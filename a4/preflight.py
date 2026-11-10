"""Checks every model with two real calls before the long run, and estimates cost.

    python a4/preflight.py --models gemini-2.5-flash-lite,gemini-3.1-flash-lite:minimal

For each model it sends (1) the cycle-1 generation prompt of Mbpp/2 and (2) the
Telugu summarization prompt for the canonical solution, prints [ok]/[FAIL], the
answers, latency and token counts, and a rough cost for the full experiment.
No response is cached, nothing is written.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.core.config import EvaluationConfig  # noqa: E402
from evaloop.models.openai_compatible import OpenAICompatibleLLM, is_api_error  # noqa: E402
from evaloop.models.vertex_genai import VertexGenAILLM  # noqa: E402
from evaloop.tasks import TaskFactory  # noqa: E402

from run_experiments import GEMINI_BASE_URL, SYSTEM_PROMPT, parse_models  # noqa: E402

# USD per 1M tokens (input, output incl. thinking); check current Gemini API pricing.
PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}
# Cycles actually run per loop configuration, from gemini-2.5-flash-lite's
# standard-loop record in the data package (sum over tasks of min(sustained+1, 10)).
CYCLES_PER_LOOP_CONFIG = 2843
N_LOOP_CONFIGS = 3


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gemini-2.5-flash-lite,gemini-3.1-flash-lite:minimal")
    parser.add_argument("--base_url", default=GEMINI_BASE_URL,
                        help='OpenAI-compatible endpoint, or "vertex" for Gemini on Vertex AI')
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    args = parser.parse_args(argv)
    if not os.environ.get(args.api_key_env) and not (
            args.base_url == "vertex" and os.environ.get("GOOGLE_CLOUD_PROJECT")):
        sys.exit(f"[FAIL] set ${args.api_key_env} (or, for --base_url vertex, $GOOGLE_CLOUD_PROJECT)")

    from evalplus.data import get_mbpp_plus

    problem = get_mbpp_plus()["Mbpp/2"]
    cfg = EvaluationConfig(model="x", output_dir="/tmp/a4_preflight", loop_type="multilingual")
    gen_task, sum_task = (TaskFactory().create_task(c) for c in cfg.get_task_configs())
    probes = [
        ("generation", gen_task.create_prompt(problem["prompt"])),
        ("telugu summary", sum_task.create_prompt(problem["canonical_solution"].strip())),
    ]

    all_ok = True
    for model, effort in parse_models(args.models):
        backend = VertexGenAILLM if args.base_url == "vertex" else OpenAICompatibleLLM
        llm = backend({
            "model_id": model, "base_url": args.base_url, "api_key_env": args.api_key_env,
            "temperature": 0.0, "top_p": 1.0, "max_tokens": 4096, "system_prompt": SYSTEM_PROMPT,
            "reasoning_effort": effort, "max_retries": 3,
        })
        try:
            for label, prompt in probes:
                start = time.time()
                text = llm.generate(prompt)
                ok = bool(text.strip()) and not is_api_error(text)
                all_ok &= ok
                print(f"[{'ok' if ok else 'FAIL'}] {model} {label} ({time.time() - start:.1f}s): "
                      f"{text.strip()[:300]}".replace("\n", " ⏎ "))
        except Exception as error:  # noqa: BLE001
            all_ok = False
            print(f"[FAIL] {model}: {type(error).__name__}: {error}")
            continue
        u = llm.usage_summary()
        calls = max(1, u["api_calls"])
        tin, tout = u["prompt_tokens"] / calls, u["completion_tokens"] / calls
        print(f"       tokens/call: in {tin:.0f}, out {tout:.0f} (thinking {u['reasoning_tokens'] / calls:.0f}); "
              f"reasoning_effort={u['reasoning_effort']}"
              f"{' (rejected, dropped)' if u['reasoning_effort_dropped'] else ''}; finish={u['finish_reasons']}")
        price = PRICES.get(model)
        if price:
            n_calls = N_LOOP_CONFIGS * 2 * CYCLES_PER_LOOP_CONFIG + 378
            cost = n_calls * (tin * price[0] + tout * price[1]) / 1e6
            print(f"       rough full-run estimate: ~{n_calls} calls, ~${cost:.2f} (summaries in Telugu "
                  "use more tokens than this average, so treat it as a lower bound)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
