"""Turns the results JSONs into summary.csv and the analysis tables.

    python a4/summarize.py --results runs/ --out analysis/ \
        [--data_package path/to/assignment2-data]

Writes to --out:
  summary.csv            the required table (model,configuration,n_tasks,success_rate,asl_score)
  summary_extended.csv   + mean loops, pass@10, pass drop, API errors, Telugu-script rate,
                         fixed points, tokens
  survival.csv / .png    pass@k for k = 1..10 per (model, configuration)
  paired.csv             per-task paired comparisons between configurations (Wilcoxon)
  probe_crosstab.csv     sustained loops split by whether the assert line alone solves the task
  reproduction.csv       our standard run vs the data package record (same model)
  report_numbers.md      everything above as markdown, for the report

success_rate = pass@1 / T (leaderboard ``successRate``);
asl_score    = sum_i n_i i^2 / (T M) (leaderboard ``aslScore`` = the framework's ASL_pow).
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.metrics import asl_pow, pass_at_k, pass_drop  # noqa: E402
from evaloop.models.openai_compatible import is_api_error  # noqa: E402

ORDER = ["standard-mbpp-plus", "english-control", "telugu-summary", "assert-only-probe"]
LOOP_CONFIGS = ORDER[:3]
TELUGU = re.compile(r"[ఀ-౿]")
LATIN = re.compile(r"[A-Za-z]")


# ----------------------------------------------------------------- loading
def load_runs(results_dir: Path):
    runs = {}
    for path in sorted(results_dir.glob("*/*_code_generation_summarization_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))[0]
        model = result.get("config", {}).get("model") or path.parent.name
        configuration = result.get("configuration") or path.name.split("_code_generation")[0]
        runs[(model, configuration)] = result
    return runs


def carried_summary(wrapped: str) -> str:
    """The model-written part of a wrapped summary ('\"\"\"', summary..., assert, '\"\"\"')."""
    lines = (wrapped or "").splitlines()
    return "\n".join(lines[1:-2]).strip() if len(lines) >= 3 else (wrapped or "").strip()


def telugu_ratio(text: str) -> float:
    """Share of Telugu letters among letters, ignoring `code spans` and identifiers."""
    text = re.sub(r"`[^`]*`", " ", text)
    text = " ".join(w for w in text.split() if not re.search(r"[_()\[\]=.0-9]", w))
    te, la = len(TELUGU.findall(text)), len(LATIN.findall(text))
    return te / (te + la) if te + la else 0.0


def cycle_flags(prompt_result):
    """API-error and fixed-point information for one task loop."""
    api_error_cycle = None
    fixed_point_cycle = None
    previous = None
    for cycle in prompt_result["cycles"]:
        raws = [cycle.get(k) for k in ("task_1_raw_output", "task_2_raw_output")]
        if api_error_cycle is None and any(is_api_error(r) for r in raws if r):
            api_error_cycle = int(cycle["cycle"])
        summary = cycle.get("task_2_output")
        if fixed_point_cycle is None and summary and previous == summary and cycle.get("success") in (True, "True"):
            fixed_point_cycle = int(cycle["cycle"])
        previous = summary
    return api_error_cycle, fixed_point_cycle


def describe(model, configuration, result):
    m = int(result.get("max_cycles", 10))
    loops = [int(r["successful_cycles"]) for r in result["prompt_results"]]
    n = len(loops)
    row = {
        "model": model, "configuration": configuration, "n_tasks": n,
        "success_rate": round(pass_at_k(loops, 1) / n, 3),
        "asl_score": round(asl_pow(loops, m), 3),
        "max_cycles": m,
        "mean_loops": round(mean(loops), 3),
        "pass_at_10": round(pass_at_k(loops, m) / n, 3) if m == 10 else "",
        "pass_drop_pp": round(100 * pass_drop(loops, m), 2),
    }
    api_tasks = 0
    api_ended = 0
    fixed = 0
    ratios = []
    for r in result["prompt_results"]:
        api_cycle, fp = cycle_flags(r)
        if api_cycle is not None:
            api_tasks += 1
            if api_cycle == r["successful_cycles"] + 1:
                api_ended += 1
        if fp is not None and r["successful_cycles"] == m:
            fixed += 1
        if configuration == "telugu-summary":
            for c in r["cycles"]:
                if c.get("success") in (True, "True") and c.get("task_2_output"):
                    ratios.append(telugu_ratio(carried_summary(c["task_2_output"])))
    usage = result.get("usage", {})
    row.update(
        api_error_tasks=api_tasks,
        loops_ended_by_api_error=api_ended,
        fixed_point_10loop_tasks=fixed,
        telugu_summaries=len(ratios),
        telugu_script_rate=round(sum(x >= 0.5 for x in ratios) / len(ratios), 3) if ratios else "",
        median_telugu_ratio=round(median(ratios), 3) if ratios else "",
        api_calls=usage.get("api_calls", ""),
        cache_hits=usage.get("cache_hits", ""),
        prompt_tokens=usage.get("prompt_tokens", ""),
        completion_tokens=usage.get("completion_tokens", ""),
        reasoning_tokens=usage.get("reasoning_tokens", ""),
        reasoning_effort=usage.get("reasoning_effort", ""),
    )
    return row, loops


# ----------------------------------------------------------------- stats
def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w

        diffs = [x - y for x, y in zip(a, b)]
        if not any(diffs):
            return 1.0
        return float(w(a, b, zero_method="wilcox").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def paired(model, a_name, a, b_name, b):
    a_map = {r["task_id"]: r["successful_cycles"] for r in a["prompt_results"]}
    b_map = {r["task_id"]: r["successful_cycles"] for r in b["prompt_results"]}
    ids = sorted(set(a_map) & set(b_map))
    xa, xb = [a_map[i] for i in ids], [b_map[i] for i in ids]
    both_c1 = [i for i in ids if a_map[i] >= 1 and b_map[i] >= 1]
    return {
        "model": model, "a": a_name, "b": b_name, "n": len(ids),
        "a_mean_loops": round(mean(xa), 3), "b_mean_loops": round(mean(xb), 3),
        "a_better": sum(x > y for x, y in zip(xa, xb)), "b_better": sum(x < y for x, y in zip(xa, xb)),
        "same": sum(x == y for x, y in zip(xa, xb)),
        "cycle1_disagree": sum((a_map[i] >= 1) != (b_map[i] >= 1) for i in ids),
        "both_pass_c1": len(both_c1),
        "a_sustain10_given_c1": round(sum(a_map[i] == 10 for i in both_c1) / max(1, len(both_c1)), 3),
        "b_sustain10_given_c1": round(sum(b_map[i] == 10 for i in both_c1) / max(1, len(both_c1)), 3),
        "wilcoxon_p": round(wilcoxon(xa, xb), 4),
    }


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, rows, cols=None):
    cols = cols or list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_package", default=None, help="folder with loops/ and leaderboard.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.results))
    if not runs:
        sys.exit("no results files found")
    key = lambda k: (k[0], ORDER.index(k[1]) if k[1] in ORDER else 99)  # noqa: E731
    rows, loops_by_run = [], {}
    for k in sorted(runs, key=key):
        row, loops = describe(*k, runs[k])
        rows.append(row)
        loops_by_run[k] = loops
    write_csv(out / "summary.csv", rows, ["model", "configuration", "n_tasks", "success_rate", "asl_score"])
    write_csv(out / "summary_extended.csv", rows)

    models = sorted({m for m, _ in runs})
    # survival curves
    surv = []
    for (m, c), loops in loops_by_run.items():
        if c in LOOP_CONFIGS:
            surv.append({"model": m, "configuration": c,
                         **{f"pass@{k}": round(pass_at_k(loops, k) / len(loops), 3) for k in range(1, 11)}})
    if surv:
        write_csv(out / "survival.csv", surv)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.2), squeeze=False, sharey=True)
            styles = {"standard-mbpp-plus": "-", "english-control": "--", "telugu-summary": ":"}
            for ax, m in zip(axes[0], models):
                for s in surv:
                    if s["model"] == m:
                        ax.plot(range(1, 11), [s[f"pass@{k}"] for k in range(1, 11)], styles[s["configuration"]],
                                marker="o", ms=3, color="black", label=s["configuration"])
                ax.set_title(m, fontsize=9)
                ax.set_xlabel("cycle k")
                ax.set_xticks(range(1, 11))
            axes[0][0].set_ylabel("pass@k (share of 378 tasks)")
            axes[0][0].legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(out / "survival.png", dpi=200)
        except Exception as error:  # noqa: BLE001
            print(f"(plot skipped: {error})")

    # paired comparisons
    pairs = []
    for m in models:
        for a, b in [("telugu-summary", "english-control"), ("english-control", "standard-mbpp-plus"),
                     ("telugu-summary", "standard-mbpp-plus")]:
            if (m, a) in runs and (m, b) in runs:
                pairs.append(paired(m, a, runs[(m, a)], b, runs[(m, b)]))
    if pairs:
        write_csv(out / "paired.csv", pairs)

    # assert-only probe cross-tab
    probe_rows = []
    for m in models:
        probe = runs.get((m, "assert-only-probe"))
        if not probe:
            continue
        solved = {r["task_id"]: r["successful_cycles"] >= 1 for r in probe["prompt_results"]}
        for c in LOOP_CONFIGS:
            if (m, c) not in runs:
                continue
            res = {r["task_id"]: r["successful_cycles"] for r in runs[(m, c)]["prompt_results"]}
            for flag in (True, False):
                ids = [i for i, s in solved.items() if s == flag and i in res and res[i] >= 1]
                probe_rows.append({
                    "model": m, "configuration": c, "assert_alone_solves": flag, "n_pass_c1": len(ids),
                    "sustain_10": round(sum(res[i] == 10 for i in ids) / max(1, len(ids)), 3),
                    "mean_loops": round(mean([res[i] for i in ids]), 3) if ids else "",
                })
    if probe_rows:
        write_csv(out / "probe_crosstab.csv", probe_rows)

    # reproduction against the data package
    repro = []
    if args.data_package:
        dp = Path(args.data_package)
        for m in models:
            ours = runs.get((m, "standard-mbpp-plus"))
            path = dp / "loops" / f"{m}.json"
            if not ours or not path.exists():
                continue
            theirs = {t["task_id"]: t["successful_cycles"] for t in json.loads(path.read_text())["tasks"]}
            mine = {r["task_id"]: r["successful_cycles"] for r in ours["prompt_results"]}
            ids = sorted(set(theirs) & set(mine))
            a, b = [mine[i] for i in ids], [theirs[i] for i in ids]
            repro.append({
                "model": m, "n": len(ids),
                "ours_success_rate": round(pass_at_k(a, 1) / len(a), 3),
                "package_success_rate": round(pass_at_k(b, 1) / len(b), 3),
                "ours_asl": round(asl_pow(a, 10), 3), "package_asl": round(asl_pow(b, 10), 3),
                "cycle1_agree": round(sum((x >= 1) == (y >= 1) for x, y in zip(a, b)) / len(a), 3),
                "same_loops": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
            })
            # also put the package record in the extended table, clearly labelled
            rows.append({"model": m, "configuration": "standard-mbpp-plus (data package exp_1)",
                         "n_tasks": len(b), "success_rate": round(pass_at_k(b, 1) / len(b), 3),
                         "asl_score": round(asl_pow(b, 10), 3), "mean_loops": round(mean(b), 3)})
        if repro:
            write_csv(out / "reproduction.csv", repro)
            write_csv(out / "summary_extended.csv", rows, list(rows[0]))

    # ranking per configuration
    rank_lines = []
    for c in LOOP_CONFIGS:
        ranked = sorted((r for r in rows if r["configuration"] == c), key=lambda r: -r["asl_score"])
        if ranked:
            rank_lines.append(f"- {c}: " + " > ".join(f"{r['model']} ({r['asl_score']})" for r in ranked))

    md = ["# Assignment 4 numbers", "", "## summary", md_table(rows, list(rows[0])), ""]
    if pairs:
        md += ["## paired per-task comparisons (successful loops)", md_table(pairs, list(pairs[0])), ""]
    if probe_rows:
        md += ["## assert-only probe cross-tab (tasks passing cycle 1)", md_table(probe_rows, list(probe_rows[0])), ""]
    if repro:
        md += ["## reproduction vs data package", md_table(repro, list(repro[0])), ""]
    md += ["## ranking by asl_score", *rank_lines, ""]
    (out / "report_numbers.md").write_text("\n".join(md), encoding="utf-8")
    print((out / "report_numbers.md").read_text(encoding="utf-8"))
    hist = {f"{m}/{c}": dict(sorted(Counter(v).items())) for (m, c), v in loops_by_run.items()}
    (out / "loop_histograms.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
