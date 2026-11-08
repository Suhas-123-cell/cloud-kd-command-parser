"""Round 2 of the teacher-prompt work: reproduce results/teacher_prompt_iterations/round2/.

Run from the repo root with Ollama (phi3.5) on 127.0.0.1:11435:  .venv/bin/python scripts/teacher_prompt_round2.py

Variants (all temperature 0, seed 42, num_predict 96, stop "\\n\\n"):
  base         round-1 final prompt (prompt_round1_final.txt), free text
  base_schema  round-1 final prompt + Ollama structured output (worker/teacher.py FORMAT)
  v2           revised prompt whose place rule quoted "near me"/"around here", which occur in
               validation.jsonl (prompt_v2_leaky.txt); kept only to show the leak was removed
  v2_schema    the same + structured output
  v3_schema    final prompt (prompts/teacher_prompt.txt): v2 with the leaked phrases replaced
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path[:0] = ["worker", "tools"]
from score import score  # noqa: E402
from teacher import FORMAT, OPTIONS  # noqa: E402

D = "results/teacher_prompt_iterations"
OUT = f"{D}/round2"
RUNS = {
    "base": (f"{D}/prompt_round1_final.txt", False),
    "base_schema": (f"{D}/prompt_round1_final.txt", True),
    "v2": (f"{OUT}/prompt_v2_leaky.txt", False),
    "v2_schema": (f"{OUT}/prompt_v2_leaky.txt", True),
    "v3_schema": ("prompts/teacher_prompt.txt", True),
}
gold = [json.loads(line) for line in open("data/validation.jsonl")]
summary = {}
for name, (prompt_file, constrained) in RUNS.items():
    tpl = open(prompt_file).read()

    def call(r, tpl=tpl, constrained=constrained):
        body = {"model": "phi3.5", "prompt": tpl.replace("{command}", r["text"]), "stream": False,
                "options": OPTIONS}
        if constrained:
            body["format"] = FORMAT
        return requests.post("http://127.0.0.1:11435/api/generate", json=body, timeout=300).json()["response"]

    with ThreadPoolExecutor(2) as ex:
        preds = [{"id": g["id"], "prediction": o} for g, o in zip(gold, ex.map(call, gold))]
    with open(f"{OUT}/{name}_predictions.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in preds)
    summary[name] = score(preds, gold)
    with open(f"{OUT}/{name}_score.json", "w") as f:
        json.dump(summary[name], f, indent=2)
    print(name, summary[name])
json.dump(summary, open(f"{OUT}/scores.json", "w"), indent=2)
