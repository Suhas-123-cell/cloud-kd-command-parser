"""Run the teacher on a labelled file and write predictions in score.py format.

python worker/eval_teacher.py data/validation.jsonl results/teacher_validation_predictions.jsonl
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from teacher import ask_teacher, load_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("out")
    ap.add_argument("--url", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    template = load_prompt(a.prompt)
    rows = [json.loads(line) for line in open(a.gold, encoding="utf-8") if line.strip()]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        replies = list(pool.map(lambda r: ask_teacher(template, r["text"], a.url), rows))
    with open(a.out, "w", encoding="utf-8") as f:
        for r, reply in zip(rows, replies):
            f.write(json.dumps({"id": r["id"], "prediction": reply}) + "\n")
    print(f"{len(rows)} commands in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
