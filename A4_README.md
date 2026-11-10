# Assignment 4 extension: the Telugu-summary loop (Option B, new attack)

CSC 591/791 Trustworthy AI for SE, Fall 2026 — Ishwarya Gandamsetty.
Fork of [evalooop/EvaLooop](https://github.com/evalooop/EvaLooop) at upstream commit `ce22fe2`,
branch `a4-telugu-loop`. Public fork: https://github.com/IG-08/EvaLooop/tree/a4-telugu-loop

## 1. What the extension is

The standard EvaLoop loop is *code → one English sentence → code → …*, with the original MBPP+
assert line re-attached to every summary. This fork adds a loop type in which the carried
summary is a one-sentence instruction written in **Telugu**, a lower-resource language with its
own script. The model's own output remains the only source of perturbation: there is no
machine-translation system and no human edit anywhere in the loop.

```
cycle k:  generation prompt( """<sentence written by the model in cycle k-1>\n<ORIGINAL assert line>""" )
             └─► code ──► EvalPlus base + plus tests ──► fail: stop (task sustained k-1 loops)
                                                     └─► pass: summarize the code in Telugu ──► cycle k+1
cycle 1:  the original English MBPP+ prompt (identical in every configuration)
```

| | standard loop (upstream) | Telugu loop (this fork) |
|---|---|---|
| Generation prompt | `Generate Python code for the following task: \n{description}\nReturn only the code, without explanations or comments.` | identical |
| Code extraction | first fenced block, else whole response | identical |
| Summarization prompt | `Use one sentence to summarize the following code and start with 'write a python function to': …` | see below |
| Carried between cycles | `"""` + summary + original assert line + `"""` | same wrapper, same untranslated assert line |
| Signature stability | the original assert line (inherited `postprocess`) | the same mechanism, unchanged |
| Failure condition | generated code fails the MBPP base **or** EvalPlus plus tests | identical |
| Max cycles / decoding | 10 / greedy (temperature 0, top-p 1) | identical |

Telugu summarization prompt (`evaloop/tasks/code_summarization_multilingual.py`):

```
Use one sentence in Telugu (తెలుగు) to summarize the following code as an instruction to write a
Python function. Keep Python identifiers (function and parameter names) unchanged. Write the rest
of the sentence only in Telugu.

```
{code}
```

Return only the Telugu sentence.
```

The response is stripped; a code fence or quotes wrapping the whole answer are removed, lines are
joined into one, and `"""` is replaced by `'''` so it cannot close the docstring.

**Why the assert line is kept untranslated.** The assert line is what keeps the function name and
argument order fixed from cycle to cycle; without it a re-described task would drift in naming
and the tests could not even call the function. Translating it would break that guarantee, and
keeping it means that the only thing that changes between the standard and the Telugu loop is the
language of the natural-language description.

## 2. Configurations (one results JSON each, per model)

| configuration | loop | purpose |
|---|---|---|
| `standard-mbpp-plus` | upstream loop, re-run | baseline (required) |
| `telugu-summary` | the extension | does a Telugu description sustain the loop? |
| `english-control` | the extension's instruction text, but `language="English"` | wording control: a Telugu sentence cannot start with "write a python function to", so the instruction had to change. `telugu − english-control` isolates the language; `english-control − standard` isolates the wording. |
| `assert-only-probe` | 1 cycle, prompt = only `"""\n<assert line>\n"""` | how much of each task the assert line alone gives away (over-reliance check) |

Models: `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite` (both with thinking level `minimal`), on
Vertex AI (`--base_url vertex`, google-genai SDK), because the course credit is on a university Google
Cloud account for which AI Studio is disabled. The originally planned `gemini-2.5-flash-lite` (which is in
the data package and would have allowed a reproduction check against the provided records) was retired on
Vertex AI on Oct 20, 2026 and is closed to new Gemini-API projects; none of the 60 data-package models is
available to us, so the standard-loop baseline is re-run for both models rather than taken from the package.
All 378 MBPP+ v0.2.0 tasks, no subset.

## 3. How to run

**Colab (what was used).** Open `a4/run_a4_colab.ipynb` in Colab, set `PROJECT_ID` (Vertex AI,
sign in when asked) or switch `PROVIDER` to `gemini_api` and add a secret `GEMINI_API_KEY`, then *Run all*. It runs the tests, a preflight, all runs, and the analysis, and
saves everything to Google Drive. Re-running resumes from the cache.

**Anywhere else (Linux/macOS, Python ≥ 3.10; run it in a container or VM, since generated code is executed):**

```bash
pip install evalplus openai google-genai fire tqdm scipy pytest && pip install -e . --no-deps   # no torch/vllm
export GEMINI_API_KEY=...        # or: gcloud auth application-default login; export GOOGLE_CLOUD_PROJECT=...
python -m pytest -q                                   # 104 tests, offline
# for Vertex AI add  --base_url vertex  to preflight.py and run_experiments.py
python a4/preflight.py                                # 2 real calls per model + cost estimate
python a4/run_experiments.py --out runs/              # all models x configurations
python a4/summarize.py --results runs/ --out analysis/ --data_package <assignment2-data>
python a4/failures.py  --results runs/ --out analysis/inspection --per_cell 8 --data_package <assignment2-data>
```

The new loop is also available from the framework's own CLI:

```bash
evaloop evaluate --model gemini-2.5-flash-lite --loop_type multilingual --summary_language Telugu \
  --api_base_url https://generativelanguage.googleapis.com/v1beta/openai/ --api_key_env GEMINI_API_KEY \
  --system_prompt "You are a helpful AI assistant." --temperature 0 --max_cycles 10
```

**Offline dry run** (no API key; checks the plumbing with a fake model):
`python a4/fake_server.py & FAKE_KEY=x python a4/run_experiments.py --out /tmp/dry --base_url http://127.0.0.1:8765/v1 --api_key_env FAKE_KEY --models fake --n_tasks 40`

## 4. How each reported number is produced

* `summary.csv` (`model,configuration,n_tasks,success_rate,asl_score`) ← `a4/summarize.py` over
  `runs/<model>/<configuration>_code_generation_summarization_results.json`.
  `success_rate` = pass@1 / T (leaderboard `successRate`); `asl_score` = Σᵢ nᵢ·i² / (T·M), the
  framework's `ASL_pow` = the leaderboard's `aslScore`. For the probe, M = 1, so its `asl_score`
  equals its pass rate.
* `summary_extended.csv`, `survival.csv/.png`, `paired.csv` (per-task Wilcoxon signed-rank tests),
  `probe_crosstab.csv`, `reproduction.csv` (our standard run vs. the data package record of
  `gemini-2.5-flash-lite`) ← same script.
* Manual inspection ← `a4/failures.py` samples, with a fixed seed, failed loops that broke at cycle ≥ 2
  per (model, configuration) and from the data package; labels are in `labels.csv`.
* Token counts and API errors per run are stored under `usage` in every results JSON.
* No LLM judge is used, so there is no semantic-similarity weighting; the headline score is `ASL_pow`.

## 5. What was changed in the framework, and why

New files:

* `evaloop/tasks/code_summarization_multilingual.py` — the new summarization task
  (`@register_task("code_summarization_multilingual")`); inherits the assert-line `postprocess`.
* `evaloop/models/openai_compatible.py` — back-end for OpenAI-compatible endpoints (Gemini). It
  never returns an API error as model text (see below), runs `batch_generate` concurrently, keeps
  a resumable JSONL response cache, and records token usage.
* `evaloop/models/vertex_genai.py` — the same back-end over Vertex AI (google-genai SDK; API key or
  Application Default Credentials); selected with `api_base_url="vertex"`.
* `a4/` — experiment driver, preflight, analysis, failure sampler, fake server, Colab notebook.
* `tests/test_multilingual_loop.py` — 28 offline tests (76 upstream + 28 new = 104).

Modified files (small):

* `evaloop/core/config.py` — `loop_type` / `summary_language` options; `get_task_configs` returns
  the upstream pair unchanged for `standard` and swaps only the summarization step for
  `multilingual`; `api_base_url` selects the new back-end; `to_dict` records the loop.
* `evaloop/core/evaluator.py` — the results file stores the run config and the back-end's usage.
* `evaloop/cli.py` — exposes the new options on `evaloop evaluate`.
* `evaloop/models/factory.py` — registers `openai_compatible` and `vertex_genai`; back-ends are imported lazily so that
  API-only runs do not need torch/vllm installed.
* `evaloop/tasks/__init__.py` — imports the new task so it registers.

The driver `a4/run_experiments.py` subclasses `EvaLoopEvaluator` only to (1) build the
assert-only probe prompts and an optional seeded subset, and (2) tag the configuration name.

## 6. Things found while reading the code (relevant to the analysis)

1. **Summarization prompt mismatch.** The data package's `prompts/task_prompts.md` documents a
   "detailed description" prompt, but the summaries in the data are one sentence starting with
   "write a python function to"; those match the repo's `get_task_configs` prompt, which is the
   one our `standard-mbpp-plus` run uses.
2. **API errors scored as model failures upstream.** `OpenAILLM.generate` returns strings such as
   `Error generating response: …` as if the model wrote them; the loop then extracts "code" from
   that text, it fails the tests, and the task's loop ends. The new back-end returns a marked
   sentinel instead, and `summarize.py` reports `api_error_tasks` / `loops_ended_by_api_error`.
3. **Both steps run before testing.** In each cycle the runner generates code *and* its summary for
   every active task before testing, so summaries of failing code are produced but never used.
   Left unchanged (costs ~one extra call per failure).
4. **Determinism.** With the response cache, an identical prompt always gets the identical answer
   within a model, which makes greedy decoding exactly deterministic across configurations: cycle 1
   is identical in the three loop configurations, and a loop whose summary repeats itself has
   reached a fixed point and will sustain all 10 cycles (`fixed_point_10loop_tasks`).

## 7. Safety and cost

Generated code is executed only by EvalPlus (`untrusted_check`, with its resource limits) inside
the Colab VM, never on a personal machine. Exact token usage per run is in each results JSON
(`usage`); the cost is computed from those counts and the Gemini API price list in the report.
