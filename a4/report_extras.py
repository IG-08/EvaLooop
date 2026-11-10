"""Extra numbers used in the report, beyond summarize.py.

    python a4/report_extras.py --results runs/ --models gemini-3.1-flash-lite,gemini-3.5-flash-lite \
        [--data_package path/to/assignment2-data]

Prints, per (model, loop configuration):
  * broken loops (failed at cycle >= 2) and the cycle at which they broke,
  * base-vs-plus failure split, failures whose code is identical to the code that just
    passed (test-harness flakiness, e.g. time limits) and failures without the entry point,
  * exact fixed points (summary of cycle k == summary of cycle k-1) and when they occur,
  * median length of the carried summaries,
  * share of cycle-1 summaries that state an edge case or bound (lexical pattern, English + Telugu),
  * tasks broken only in Telugu (English control sustained all 10 cycles), and vice versa,
and, with --data_package, the fixed-point share of 10-loop tasks for the 60 packaged models.
"""

import argparse
import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize import carried_summary, load_runs  # noqa: E402

EDGE = re.compile(r"None|inf|float\('inf'\)|empty|ఖాళీ|0\.0|-1\b|otherwise|లేకపోతే|strictly|at least|"
                  r"or equal|సమానం|కంటే తక్కువ కాని|కనీసం")
_ws = re.compile(r"\s+")


def _same(a, b):
    return _ws.sub("", a or "") == _ws.sub("", b or "")


def per_run(model, configuration, result, problems):
    broken, fail_cycle, base_fail, plus_only, identical, no_entry = 0, [], 0, 0, [], []
    fixed, lengths, edge, n_c1 = [], [], 0, 0
    max_cycles = int(result.get("max_cycles", 10))
    for p in result["prompt_results"]:
        k = p["successful_cycles"]
        previous = None
        for cycle in p["cycles"]:
            summary = cycle.get("task_2_output")
            ok = cycle.get("success") in (True, "True")
            if ok and summary:
                lengths.append(len(carried_summary(summary)))
            if previous is not None and ok and summary == previous:
                fixed.append(int(cycle["cycle"]))
                break
            previous = summary
        if k >= 1 and p["cycles"][0].get("task_2_output"):
            n_c1 += 1
            edge += bool(EDGE.search(carried_summary(p["cycles"][0]["task_2_output"])))
        if 1 <= k < max_cycles:
            broken += 1
            fail_cycle.append(k + 1)
            fail, prev = p["cycles"][k], p["cycles"][k - 1]
            tests = fail.get("test_results") or {}
            if tests.get("base_success") is False:
                base_fail += 1
            elif tests.get("plus_success") is False:
                plus_only += 1
            if _same(fail.get("task_1_output"), prev.get("task_1_output")):
                identical.append(p["task_id"])
            entry = problems[p["task_id"]]["entry_point"]
            if f"def {entry}(" not in (fail.get("task_1_output") or ""):
                no_entry.append(p["task_id"])
    ten = sum(1 for p in result["prompt_results"] if p["successful_cycles"] == max_cycles)
    return {
        "broken_loops": broken,
        "broke_at_cycle": dict(sorted(collections.Counter(fail_cycle).items())),
        "base_fail": base_fail, "plus_only_fail": plus_only,
        "identical_code_failures": identical, "missing_entry_point": no_entry,
        "ten_cycle_tasks": ten, "fixed_points": len(fixed),
        "fixed_point_at_cycle": dict(sorted(collections.Counter(fixed).items())),
        "median_summary_chars": statistics.median(lengths) if lengths else None,
        "cycle1_edge_case_share": round(edge / n_c1, 3) if n_c1 else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--models", default=None)
    parser.add_argument("--data_package", default=None)
    parser.add_argument("--out", default=None, help="optional JSON output file")
    args = parser.parse_args(argv)
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    runs = load_runs(Path(args.results), args.models)
    report = {}
    for (model, configuration), result in sorted(runs.items()):
        if result.get("probe"):
            continue
        report[f"{model}/{configuration}"] = per_run(model, configuration, result, problems)
    for model in sorted({m for m, _ in runs}):
        if (model, "telugu-summary") not in runs or (model, "english-control") not in runs:
            continue
        te = {p["task_id"]: p["successful_cycles"] for p in runs[(model, "telugu-summary")]["prompt_results"]}
        en = {p["task_id"]: p["successful_cycles"] for p in runs[(model, "english-control")]["prompt_results"]}
        report[f"{model}/telugu-vs-control"] = {
            "telugu_broken_control_sustained": sorted(t for t in te if 1 <= te[t] < 10 and en[t] == 10),
            "control_broken_telugu_sustained": sorted(t for t in te if 1 <= en[t] < 10 and te[t] == 10),
            "both_broken": sorted(t for t in te if 1 <= en[t] < 10 and 1 <= te[t] < 10),
        }
    if args.data_package:
        total = fixed = 0
        for path in glob.glob(str(Path(args.data_package) / "loops" / "*.json")):
            for task in json.loads(Path(path).read_text())["tasks"]:
                if task["successful_cycles"] == 10:
                    total += 1
                    summaries = [c.get("summary") for c in task["cycles"]]
                    fixed += any(summaries[i] and summaries[i] == summaries[i - 1] for i in range(1, len(summaries)))
        report["data_package_10_loop_fixed_points"] = {"tasks": total, "fixed": fixed,
                                                      "share": round(fixed / total, 3) if total else None}
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
