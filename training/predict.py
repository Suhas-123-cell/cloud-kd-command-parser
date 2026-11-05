"""Run a (student or base) model over a labelled file and write predictions for score.py.

python training/predict.py MODEL_DIR_OR_HF_ID data/validation.jsonl results/student_v1_validation_predictions.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))
from inference import Parser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("gold")
    ap.add_argument("out")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    a = ap.parse_args()
    parser = Parser(a.model, max_new_tokens=a.max_new_tokens)
    rows = [json.loads(ln) for ln in open(a.gold, encoding="utf-8") if ln.strip()]
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prediction": parser.generate(r["text"])}) + "\n")
    print(f"{len(rows)} predictions in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
