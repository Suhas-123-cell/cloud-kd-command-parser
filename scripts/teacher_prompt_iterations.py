"""Save evidence for the teacher-prompt iterations (run from the repo root with the venv python)."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path[:0] = ["worker", "tools"]
from score import score  # noqa: E402

OUT = "results/teacher_prompt_iterations"
os.makedirs(OUT, exist_ok=True)
final = open("prompts/teacher_prompt.txt").read()
draft = final.replace(" A person's name is never a place.", "").replace(
    """4. Every slot value must be words copied from the command, or null.

Write only the JSON object on one line. Do not add notes, explanations or more examples after it.
""", "")
assert draft != final
open(f"{OUT}/prompt_v0_first_draft.txt", "w").write(draft)
gold = [json.loads(line) for line in open("data/validation.jsonl")]

runs = {
    "v0_first_draft_no_stop": (draft, {}),
    "v1_final_prompt_no_stop": (final, {}),
    "v1_final_prompt_stop_blankline": (final, {"stop": ["\n\n"]}),
}
summary = {}
for name, (tpl, extra) in runs.items():
    def call(r, tpl=tpl, extra=extra):
        body = {"model": "phi3.5", "prompt": tpl.replace("{command}", r["text"]), "stream": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": 96, **extra}}
        return requests.post("http://127.0.0.1:11435/api/generate", json=body, timeout=300).json()["response"]
    with ThreadPoolExecutor(2) as ex:
        outs = list(ex.map(call, gold))
    preds = [{"id": g["id"], "prediction": o} for g, o in zip(gold, outs)]
    with open(f"{OUT}/{name}_predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    summary[name] = score(preds, gold)
    print(name, summary[name])
json.dump(summary, open(f"{OUT}/scores.json", "w"), indent=2)
