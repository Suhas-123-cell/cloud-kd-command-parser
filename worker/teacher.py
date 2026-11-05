"""Call the teacher model (phi3.5 served by Ollama) with the fixed prompt and settings.

The reply is returned exactly as generated: no stripping, no repair.
"""
import os
from pathlib import Path

import requests

TEACHER_MODEL = "phi3.5"
# Fixed by the assignment: temperature 0, seed 42.
# num_predict caps runaway replies; the blank-line stop sequence ends generation before
# phi3.5 appends "Note: ..." paragraphs or invents extra examples (validity 89% -> 99% on validation).
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 96, "stop": ["\n\n"]}
DEFAULT_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "teacher_prompt.txt"


def load_prompt(path=None):
    return Path(path or os.environ.get("PROMPT_FILE", DEFAULT_PROMPT)).read_text(encoding="utf-8")


def build_prompt(template, command):
    # str.replace, not str.format: the template contains literal JSON braces.
    return template.replace("{command}", command)


def ask_teacher(template, command, url=None, timeout=300):
    url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    resp = requests.post(
        f"{url}/api/generate",
        json={
            "model": TEACHER_MODEL,
            "prompt": build_prompt(template, command),
            "stream": False,
            "options": OPTIONS,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]
