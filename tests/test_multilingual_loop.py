"""Tests for the Assignment 4 extension: multilingual summary loop + OpenAI-compatible back-end."""

import json
from types import SimpleNamespace

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.models.factory import ModelFactory
from evaloop.models.openai_compatible import (
    API_ERROR_PREFIX,
    FatalAPIError,
    OpenAICompatibleLLM,
    is_api_error,
)
from evaloop.tasks import MultilingualCodeSummarizationTask, TaskFactory

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT, FakeLLM

TELUGU = "`similar_elements` అనే పైథాన్ ఫంక్షన్‌ను రాయండి."


# ---------------------------------------------------------------- task
class TestMultilingualTask:
    def test_registered(self):
        task = TaskFactory().create_task({"type": "code_summarization_multilingual", "language": "Telugu"})
        assert isinstance(task, MultilingualCodeSummarizationTask)
        assert task.produces_testable_code is False

    def test_prompt_names_language_and_embeds_code(self):
        task = MultilingualCodeSummarizationTask({"language": "Telugu"})
        prompt = task.create_prompt("def f(x): return x")
        assert "Telugu (తెలుగు)" in prompt
        assert "def f(x): return x" in prompt
        assert "write a python function to" not in prompt

    def test_english_control_uses_same_wording(self):
        te = MultilingualCodeSummarizationTask({"language": "Telugu"}).create_prompt("CODE")
        en = MultilingualCodeSummarizationTask({"language": "English"}).create_prompt("CODE")
        assert en == te.replace("Telugu (తెలుగు)", "English").replace("Telugu", "English")

    @pytest.mark.parametrize("raw", [
        TELUGU, f"  {TELUGU}\n", f"```\n{TELUGU}\n```", f"```text\n{TELUGU}\n```", f'"{TELUGU}"',
    ])
    def test_extract_unwraps(self, raw):
        assert MultilingualCodeSummarizationTask({}).extract_output(raw) == TELUGU

    def test_extract_joins_lines_and_neutralises_docstring_quotes(self):
        out = MultilingualCodeSummarizationTask({}).extract_output('line one\n\n"""line two"""')
        assert out == "line one '''line two'''"

    def test_postprocess_keeps_original_assert_line(self):
        task = MultilingualCodeSummarizationTask({"language": "Telugu"})
        wrapped = task.postprocess(TELUGU, {"cycle_input": MBPP_STYLE_PROMPT})
        assert wrapped == f'"""\n{TELUGU}\n{ASSERT_LINE}\n"""\n'
        # and the next cycle (whose input is this wrapped text) keeps the same assert
        again = task.postprocess("x", {"cycle_input": wrapped})
        assert again.splitlines()[-2] == ASSERT_LINE


# ---------------------------------------------------------------- config
class TestConfig:
    def test_standard_unchanged(self, tmp_path):
        tasks = EvaluationConfig(model="m", output_dir=str(tmp_path)).get_task_configs()
        assert [t["type"] for t in tasks] == ["code_generation", "code_summarization"]
        assert "write a python function to" in tasks[1]["prompt_template"]

    def test_multilingual_swaps_only_summarization(self, tmp_path):
        std = EvaluationConfig(model="m", output_dir=str(tmp_path)).get_task_configs()
        ml = EvaluationConfig(model="m", output_dir=str(tmp_path), loop_type="multilingual",
                              summary_language="Telugu").get_task_configs()
        assert ml[0] == std[0]
        assert ml[1] == {"type": "code_summarization_multilingual", "language": "Telugu"}

    def test_bad_loop_type(self, tmp_path):
        with pytest.raises(ValueError):
            EvaluationConfig(model="m", output_dir=str(tmp_path), loop_type="nope")

    def test_base_url_selects_openai_compatible(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        cfg = EvaluationConfig(model="gemini-2.5-flash-lite", output_dir=str(tmp_path),
                               api_base_url="http://h/v1", api_key_env="MY_KEY")
        model_cfg = cfg.get_model_config()
        assert model_cfg["type"] == "openai_compatible"
        assert model_cfg["api_key"] == "secret"
        assert cfg.to_dict()["loop_type"] == "standard"


# ---------------------------------------------------------------- back-end
def _response(text, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, completion_tokens_details=None),
    )


class ScriptedClient:
    """Mimics client.chat.completions.create with a scripted list of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _response(outcome)


def _error(name, message="boom"):
    return type(name, (Exception,), {})(message)


def _llm(outcomes, **extra):
    config = {"model_id": "m", "max_retries": 2, **extra}
    return OpenAICompatibleLLM(config, client=ScriptedClient(outcomes))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("evaloop.models.openai_compatible.time.sleep", lambda s: None)


class TestOpenAICompatibleLLM:
    def test_greedy_request(self):
        llm = _llm(["hi"], system_prompt="sys")
        assert llm.generate("p") == "hi"
        call = llm.client.calls[0]
        assert call["temperature"] == 0.0 and call["top_p"] == 1.0
        assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "p"}]
        assert "reasoning_effort" not in call
        assert llm.usage_summary()["prompt_tokens"] == 10

    def test_retries_then_succeeds(self):
        llm = _llm([_error("RateLimitError"), "ok"])
        assert llm.generate("p") == "ok"
        assert llm.usage["retries"] == 1 and llm.usage["api_errors"] == 0

    def test_persistent_failure_returns_sentinel_not_text(self):
        llm = _llm([_error("RateLimitError")] * 3)
        out = llm.generate("p")
        assert out.startswith(API_ERROR_PREFIX) and is_api_error(out)
        assert llm.usage["api_errors"] == 1

    def test_fatal_errors_abort(self):
        with pytest.raises(FatalAPIError):
            _llm([_error("AuthenticationError")]).generate("p")

    def test_reasoning_effort_dropped_if_rejected(self):
        llm = _llm([_error("BadRequestError", "reasoning_effort not supported"), "ok"],
                   reasoning_effort="minimal")
        assert llm.generate("p") == "ok"
        assert "reasoning_effort" in llm.client.calls[0]
        assert "reasoning_effort" not in llm.client.calls[1]
        assert llm.usage["reasoning_effort_dropped"] is True

    def test_empty_response_retried_then_kept(self):
        llm = _llm(["", "", ""], empty_retries=2)
        assert llm.generate("p") == ""
        assert len(llm.client.calls) == 3 and llm.usage["empty_responses"] == 1

    def test_cache_replays_and_skips_errors(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        first = _llm(["a", _error("RateLimitError")] + [_error("RateLimitError")] * 2, cache_path=str(cache))
        assert first.generate("p1") == "a"
        assert is_api_error(first.generate("p2"))
        assert len(cache.read_text().splitlines()) == 1  # error not cached
        second = _llm(["b"], cache_path=str(cache))
        assert second.generate("p1") == "a" and second.usage["cache_hits"] == 1
        assert second.generate("p2") == "b"  # retried for real

    def test_batch_generate_keeps_order(self):
        llm = _llm([f"r{i}" for i in range(20)], max_concurrency=1)
        assert llm.batch_generate([f"p{i}" for i in range(20)]) == [f"r{i}" for i in range(20)]

    def test_factory(self):
        model = ModelFactory().create_model({"type": "openai_compatible", "model_id": "m", "api_key": "k",
                                             "base_url": "http://127.0.0.1:9/v1"})
        assert isinstance(model, OpenAICompatibleLLM)


# ---------------------------------------------------------------- loop integration
class EchoLLM(FakeLLM):
    """Returns fixed code for generation prompts and a Telugu sentence otherwise."""

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if prompt.startswith("Generate Python code"):
            return "```python\ndef similar_elements(a, b):\n    return tuple(set(a) & set(b))\n```"
        return TELUGU


def test_one_cycle_carries_telugu_summary_with_assert(generation_config):
    tasks = [TaskFactory().create_task(generation_config),
             TaskFactory().create_task({"type": "code_summarization_multilingual", "language": "Telugu"})]
    llm = EchoLLM()
    out = BatchEvaluationCycle(llm, tasks).run_single_cycle(
        [{"task_id": "Mbpp/2", "prompt": MBPP_STYLE_PROMPT}], 1)["cycle_results"][0]
    assert out["task_2_output"] == f'"""\n{TELUGU}\n{ASSERT_LINE}\n"""\n'
    assert "Telugu (తెలుగు)" in llm.prompts[1]
    json.dumps(out, ensure_ascii=False)  # serialisable
