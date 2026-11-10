"""Samples failed loops for manual inspection and writes a labelling sheet.

    python a4/failures.py --results runs/ --out analysis/inspection \
        [--per_cell 8] [--data_package path/to/assignment2-data]

Only loops that broke at cycle >= 2 are sampled: cycle 1 is the original MBPP+
prompt in every configuration, so a cycle-1 failure says nothing about the loop.
For each sampled loop the markdown shows the text that was carried into the
failing cycle (for telugu-summary: the Telugu sentence), the code that passed
before it, the failing code and the test outcome, plus automatic hints. The CSV
has empty `label` / `notes` columns for the hand labels. With --data_package, the
same number of failed loops of the same model is sampled from the provided
baseline records for comparison.

Suggested labels (several may apply; the first is the dominant cause):
  REQ_DROPPED      the summary omits a requirement / edge case of the original task
  REQ_CHANGED      the summary states a different requirement (mistranslation, wrong term)
  IMPL_MISMATCH    the summary is adequate but the code does not implement it
  ASSERT_RELIANCE  the code fits the single assert example but not the task
  LANG_DRIFT       the carried text is not (mainly) in the requested language
  FORMAT           extraction / syntax / missing entry point
  API              the failure was caused by an API error, not the model
  OTHER
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize import carried_summary, load_runs, telugu_ratio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaloop.models.openai_compatible import is_api_error  # noqa: E402


def _passed(cycle):
    return cycle.get("success") in (True, "True")


def hints(code, entry_point, carried, configuration, test_results, raw_outputs):
    out = []
    if any(is_api_error(r) for r in raw_outputs if r):
        out.append("API error in this cycle")
    try:
        compile(code or "", "<generated>", "exec")
    except SyntaxError:
        out.append("does not compile")
    if entry_point and f"def {entry_point}(" not in (code or ""):
        out.append(f"no def {entry_point}()")
    if configuration == "telugu-summary":
        ratio = telugu_ratio(carried)
        if ratio < 0.5:
            out.append(f"carried text mostly not Telugu (ratio {ratio:.2f})")
    if isinstance(test_results, dict):
        if test_results.get("base_success") and not test_results.get("plus_success"):
            out.append("base tests pass, plus tests fail (edge cases)")
        elif test_results.get("base_success") is False:
            out.append("fails original MBPP tests")
        if test_results.get("error"):
            out.append(f"tester error: {str(test_results['error'])[:80]}")
    return "; ".join(out)


def ours(results_dir, per_cell, seed, problems, models=None):
    cases = []
    for (model, configuration), result in sorted(load_runs(results_dir, models).items()):
        if result.get("probe"):
            continue
        failed = [r for r in result["prompt_results"]
                  if 1 <= r["successful_cycles"] < int(result.get("max_cycles", 10))]
        for r in random.Random(f"{seed}-{model}-{configuration}").sample(failed, min(per_cell, len(failed))):
            k = r["successful_cycles"]
            fail = r["cycles"][k]
            prev = r["cycles"][k - 1]
            carried = carried_summary(prev.get("task_2_output", ""))
            entry = problems.get(r["task_id"], {}).get("entry_point")
            cases.append(dict(
                source="ours", model=model, configuration=configuration, task_id=r["task_id"],
                fail_cycle=k + 1, original_prompt=r["initial_prompt"], carried=carried,
                carried_raw=prev.get("task_2_raw_output", ""),
                previous_code=prev.get("task_1_output", ""), failing_code=fail.get("task_1_output", ""),
                failing_raw=fail.get("task_1_raw_output", ""),
                hints=hints(fail.get("task_1_output", ""), entry, carried, configuration,
                            fail.get("test_results"),
                            [fail.get("task_1_raw_output"), prev.get("task_2_raw_output")]),
            ))
    return cases


def baseline(data_package, models, per_cell, seed, problems):
    cases = []
    for model in models:
        path = Path(data_package) / "loops" / f"{model}.json"
        if not path.exists():
            continue
        tasks = json.loads(path.read_text())["tasks"]
        failed = [t for t in tasks if 1 <= t["successful_cycles"] < 10]
        for t in random.Random(f"{seed}-{model}-package").sample(failed, min(per_cell, len(failed))):
            k = t["successful_cycles"]
            fail, prev = t["cycles"][k], t["cycles"][k - 1]
            entry = problems.get(t["task_id"], {}).get("entry_point")
            tr = {"base_success": fail.get("base_passed"), "plus_success": fail.get("plus_passed")}
            cases.append(dict(
                source="data package", model=model, configuration="standard (data package exp_1)",
                task_id=t["task_id"], fail_cycle=k + 1, original_prompt=t["initial_prompt"],
                carried=carried_summary(prev.get("summary", "")), carried_raw="",
                previous_code=prev.get("code", ""), failing_code=fail.get("code", ""), failing_raw="",
                hints=hints(fail.get("code", ""), entry, "", "standard", tr, []),
            ))
    return cases


def to_markdown(cases):
    md = ["# Failed loops for manual inspection", "",
          "Labels: REQ_DROPPED, REQ_CHANGED, IMPL_MISMATCH, ASSERT_RELIANCE, LANG_DRIFT, FORMAT, API, OTHER", ""]
    for i, c in enumerate(cases, 1):
        md += [
            f"## {i}. {c['model']} / {c['configuration']} / {c['task_id']} — broke at cycle {c['fail_cycle']}",
            f"*hints:* {c['hints'] or '—'}", "",
            "**Original MBPP+ prompt**", "```", c["original_prompt"].strip(), "```",
            f"**Carried into cycle {c['fail_cycle']}** (model-written, assert line omitted)", "```",
            c["carried"], "```",
            f"**Code that passed (cycle {c['fail_cycle'] - 1})**", "```python", (c["previous_code"] or "").strip(), "```",
            f"**Failing code (cycle {c['fail_cycle']})**", "```python", (c["failing_code"] or "").strip(), "```", "",
        ]
    return "\n".join(md)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per_cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=591)
    parser.add_argument("--data_package", default=None)
    parser.add_argument("--models", default=None)
    args = parser.parse_args(argv)
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    cases = ours(Path(args.results), args.per_cell, args.seed, problems, args.models)
    if args.data_package:
        models = sorted({c["model"] for c in cases})
        cases += baseline(args.data_package, models, args.per_cell, args.seed, problems)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "failed_loops.md").write_text(to_markdown(cases), encoding="utf-8")
    cols = ["source", "model", "configuration", "task_id", "fail_cycle", "hints", "label", "notes"]
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for c in cases:
            writer.writerow({**c, "label": "", "notes": ""})
    (out / "failed_loops.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} loops -> {out / 'failed_loops.md'}, {out / 'labels.csv'}")


if __name__ == "__main__":
    main()
