"""Offline OpenAI-compatible fake model, for dry-running the whole pipeline.

    python a4/fake_server.py --port 8765 &
    python a4/run_experiments.py --base_url http://127.0.0.1:8765/v1 --api_key_env FAKE_KEY ...

Behaviour (deterministic per prompt, so caching and resumption can be tested):

* generation prompts -> the MBPP+ canonical solution of the task whose assert line
  appears in the prompt, broken on purpose for ~12% of prompts;
* summarization prompts -> a one-line summary in the requested language
  (Telugu, English, or the standard "write a python function to ..." form);
* ~3% of requests are answered with HTTP 429 to exercise the retry path.

No model is involved; this only checks plumbing (prompts, extraction, testing,
results files, ASL, summary.csv).
"""

import argparse
import hashlib
import json
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evalplus.data import get_mbpp_plus

PROBLEMS = get_mbpp_plus()
BY_ASSERT = {p["prompt"].splitlines()[-2].strip(): p for p in PROBLEMS.values()}
RNG = random.Random(0)


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def answer(prompt: str) -> str:
    if prompt.startswith("Generate Python code"):
        problem = next((p for line in prompt.splitlines()
                        if (p := BY_ASSERT.get(line.strip()))), None)
        if problem is None:
            return "```python\ndef unknown():\n    pass\n```"
        code = problem["canonical_solution"].strip("\n")
        if _h(prompt) % 100 < 12:
            code += "\n\nraise RuntimeError('fake failure')"
        return f"```python\n{code}\n```"
    names = re.findall(r"def (\w+)\(", prompt)
    name = names[0] if names else "f"
    if "in Telugu" in prompt:
        return f"`{name}` అనే పైథాన్ ఫంక్షన్‌ను రాయండి, ఇది ఇచ్చిన ఇన్‌పుట్‌ల నుండి ఫలితాన్ని లెక్కిస్తుంది."
    if "in English" in prompt:
        return f"Write a Python function `{name}` that computes the result from the given inputs."
    return f"write a python function to compute the result of {name} from the given inputs."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if RNG.random() < 0.03:
            return self._send(429, {"error": {"message": "fake rate limit", "code": 429}})
        prompt = body["messages"][-1]["content"]
        text = answer(prompt)
        self._send(200, {
            "id": "fake", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4,
                      "total_tokens": (len(prompt) + len(text)) // 4},
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
