"""Runs every (model, configuration) of Assignment 4 and writes the results JSONs.

Configurations (all on MBPP+ v0.2.0, greedy, max 10 cycles, one run each):

    standard-mbpp-plus  upstream loop (code <-> English "write a python function to" summary)
    telugu-summary      the extension: code <-> one-sentence Telugu summary
    english-control     same instruction as telugu-summary, but in English
                        (telugu - english-control = effect of the language alone)
    assert-only-probe   one cycle whose prompt is only the original assert line
                        (how much of each task the assert line gives away)

Usage (Gemini; key in $GEMINI_API_KEY):

    python a4/run_experiments.py --out runs/ \
        --models gemini-2.5-flash-lite,gemini-3.1-flash-lite:minimal

``name:effort`` sets ``reasoning_effort`` for that model. Results go to
``<out>/<model>/<configuration>_code_generation_summarization_results.json``.
Finished runs are skipped, and every model has a response cache
(``<out>/<model>/response_cache.jsonl``), so an interrupted run can simply be
started again: answers already received are replayed instead of re-requested.
The cache is shared by the configurations of a model, so cycle 1 (whose prompts
are identical in all three loop configurations) is generated once and is
identical across them by construction.

This script uses the framework's own EvaLoopEvaluator; the subclass below only
(1) supports the assert-only probe and an optional seeded task subset and
(2) records the configuration name. Generation, extraction, EvalPlus testing,
the assert-line postprocess and the loop control are the framework's.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.core.config import EvaluationConfig  # noqa: E402
from evaloop.core.evaluator import EvaLoopEvaluator  # noqa: E402

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Same system message as the upstream OpenAI back-end (OpenAILLM.DEFAULT_SYSTEM_PROMPT).
SYSTEM_PROMPT = "You are a helpful AI assistant."

CONFIGURATIONS = {
    "standard-mbpp-plus": dict(loop_type="standard"),
    "telugu-summary": dict(loop_type="multilingual", summary_language="Telugu"),
    "english-control": dict(loop_type="multilingual", summary_language="English"),
    "assert-only-probe": dict(loop_type="standard", max_cycles=1, probe=True),
}
DEFAULT_ORDER = ["standard-mbpp-plus", "telugu-summary", "english-control", "assert-only-probe"]

class A4Evaluator(EvaLoopEvaluator):
    def __init__(self, config, configuration: str, probe: bool = False,
                 task_ids: Optional[List[str]] = None):
        super().__init__(config)
        self.configuration = configuration
        self.probe = probe
        self.task_ids = task_ids

    # (1) probe prompts and task subset -------------------------------------
    def _load_dataset(self):
        prompts, problems = super()._load_dataset()
        if self.task_ids is not None:
            wanted = set(self.task_ids)
            prompts = [p for p in prompts if p["task_id"] in wanted]
        if self.probe:
            prompts = [
                {"task_id": p["task_id"],
                 "prompt": '"""\n' + p["prompt"].splitlines()[-2] + '\n"""\n'}
                for p in prompts
            ]
        return prompts, problems

    def _create_tasks(self):
        tasks = super()._create_tasks()
        return tasks[:1] if self.probe else tasks  # probe: generation only

    # (2) bookkeeping --------------------------------------------------------
    def _finalize_model_result(self, model_result):
        super()._finalize_model_result(model_result)
        model_result["configuration"] = self.configuration
        model_result["probe"] = self.probe


# ------------------------------------------------------------------- driver
def parse_models(spec: str):
    models = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            name, _, effort = item.partition(":")
            models.append((name.strip(), effort.strip() or None))
    return models


def sample_task_ids(n_tasks: int, seed: int):
    if not n_tasks:
        return None
    from evalplus.data import get_mbpp_plus

    ids = sorted(get_mbpp_plus())
    return sorted(random.Random(seed).sample(ids, n_tasks))


def results_path(out: Path, model: str, configuration: str) -> Path:
    return out / model / f"{configuration}_code_generation_summarization_results.json"


def run_one(args, model: str, effort, configuration: str, task_ids) -> Path:
    spec = dict(CONFIGURATIONS[configuration])
    probe = spec.pop("probe", False)
    max_cycles = spec.pop("max_cycles", args.max_cycles)
    model_dir = Path(args.out) / model
    model_dir.mkdir(parents=True, exist_ok=True)
    config = EvaluationConfig(
        model=model,
        max_cycles=max_cycles,
        temperature=0.0,
        top_p=1.0,
        output_dir=str(model_dir),
        experiment_name=configuration,
        max_model_len=args.max_tokens,
        system_prompt=SYSTEM_PROMPT,
        api_base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_concurrency=args.concurrency,
        reasoning_effort=effort,
        response_cache=str(model_dir / "response_cache.jsonl"),
        **spec,
    )
    evaluator = A4Evaluator(config, configuration, probe=probe, task_ids=task_ids)
    start = time.time()
    [result] = evaluator.run_code_generation_summarization()
    path = results_path(Path(args.out), model, configuration)
    result["wall_time_s"] = round(time.time() - start, 1)
    result["task_subset"] = {"n_tasks": args.n_tasks, "seed": args.seed} if task_ids else None
    path.write_text(json.dumps([result], indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="gemini-2.5-flash-lite,gemini-3.1-flash-lite:minimal")
    parser.add_argument("--configurations", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--base_url", default=GEMINI_BASE_URL)
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_cycles", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--n_tasks", type=int, default=0, help="0 = all 378 tasks")
    parser.add_argument("--seed", type=int, default=591)
    parser.add_argument("--force", action="store_true", help="re-run finished configurations")
    parser.add_argument("--log_level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not os.environ.get(args.api_key_env):
        sys.exit(f"Set ${args.api_key_env} first.")
    task_ids = sample_task_ids(args.n_tasks, args.seed)
    configurations = [c.strip() for c in args.configurations.split(",") if c.strip()]
    unknown = set(configurations) - set(CONFIGURATIONS)
    if unknown:
        sys.exit(f"Unknown configurations: {sorted(unknown)}")

    for model, effort in parse_models(args.models):
        for configuration in configurations:
            path = results_path(Path(args.out), model, configuration)
            if path.exists() and not args.force:
                print(f"[skip] {model} / {configuration} (already done)", flush=True)
                continue
            print(f"[run ] {model} / {configuration} ...", flush=True)
            start = time.time()
            done = run_one(args, model, effort, configuration, task_ids)
            result = json.loads(done.read_text(encoding="utf-8"))[0]
            cycles = [r["successful_cycles"] for r in result["prompt_results"]]
            usage = result.get("usage", {})
            print(f"[done] {model} / {configuration}: pass@1={sum(c >= 1 for c in cycles)}/{len(cycles)}, "
                  f"mean loops={sum(cycles) / len(cycles):.2f}, api_errors={usage.get('api_errors', 0)}, "
                  f"calls={usage.get('api_calls', 0)}, cache_hits={usage.get('cache_hits', 0)}, "
                  f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
