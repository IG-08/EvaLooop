"""Code summarization into a chosen natural language (Assignment 4 extension).

The standard EvaLoop loop asks for a one-sentence English summary that starts
with "write a python function to". This task asks for the same one-sentence
instruction-style summary, but written in an arbitrary natural language
(e.g. Telugu). Everything else about the loop is unchanged:

* the generation prompt (English instruction text) is reused as is;
* the summary is wrapped as a docstring together with the ORIGINAL MBPP+ assert
  line by the inherited ``postprocess``, so the function name and signature are
  kept stable across cycles exactly as in the standard loop;
* the model's own output is the only source of perturbation (no machine
  translation system and no human edits are involved).

The same class run with ``language="English"`` is the wording control: it uses
this task's instruction text, but in English, so that
``telugu - english_control`` isolates the effect of the summary language from
the effect of the changed instruction wording.
"""

import re
from typing import Any, Dict

from .code_summarization import CodeSummarizationTask
from .factory import register_task

#: Native script names shown next to the language name in the instruction.
NATIVE_NAMES = {
    "telugu": "తెలుగు",
    "hindi": "हिन्दी",
    "tamil": "தமிழ்",
    "kannada": "ಕನ್ನಡ",
}

DEFAULT_TEMPLATE = (
    "Use one sentence in {language_label} to summarize the following code as an "
    "instruction to write a Python function. Keep Python identifiers (function and "
    "parameter names) unchanged. Write the rest of the sentence only in {language}.\n\n"
    "```\n{code}\n```\n\n"
    "Return only the {language} sentence."
)

_FENCE = re.compile(r"^```[^\n]*\n([\s\S]*?)\n?```$")


@register_task("code_summarization_multilingual")
class MultilingualCodeSummarizationTask(CodeSummarizationTask):
    """One-sentence code summary in ``task_config["language"]``."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__(task_config)
        self.language = str(task_config.get("language", "Telugu")).strip()
        native = NATIVE_NAMES.get(self.language.lower())
        self.language_label = f"{self.language} ({native})" if native else self.language

    def create_prompt(self, input_text: str) -> str:
        template = self.task_config.get("prompt_template", DEFAULT_TEMPLATE)
        return template.format(
            code=input_text, language=self.language, language_label=self.language_label
        )

    def extract_output(self, raw_response: str) -> str:
        """Strip the response and undo wrappers that would corrupt the docstring.

        Kept deliberately minimal so that the carried text is the model's own:
        a code fence or quotes wrapping the whole answer are removed, newlines are
        collapsed (the standard loop's summaries are single lines), and triple
        double quotes are replaced because they would close the docstring that
        ``postprocess`` builds.
        """
        text = raw_response.strip()
        match = _FENCE.match(text)
        if match:
            text = match.group(1).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
            text = text[1:-1].strip()
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return text.replace('"""', "'''")
