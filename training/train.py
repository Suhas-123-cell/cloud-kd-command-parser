"""Sequence-level knowledge distillation: fine-tune pythia-70m on the teacher's JSON replies.

python training/train.py --config training/config.yaml [--subset 50] [--output-dir DIR] [--seed N] [--epochs N]

Reads every hyper-parameter from the config file (CLI flags only override, and are recorded
in version.json). Writes the model + tokenizer, version.json, model.sha256 (of the release
tarball) and training/loss.csv (mean training loss per epoch).
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))
from inference import training_text  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_records(cfg):
    path = ROOT / cfg["train_file"]
    if not path.exists():  # fall back to object storage
        sys.path.insert(0, str(ROOT / "scripts"))
        from s3 import client, split
        path.parent.mkdir(parents=True, exist_ok=True)
        client().download_file(*split(cfg["train_uri"]), str(path))
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def encode(tok, rows, max_len):
    examples = []
    for r in rows:
        prompt, target = training_text(r["text"], r["teacher_raw"])
        p_ids = tok(prompt).input_ids
        t_ids = tok(target).input_ids + [tok.eos_token_id]
        ids = (p_ids + t_ids)[:max_len]
        labels = ([-100] * len(p_ids) + t_ids)[:max_len]  # loss on the teacher reply only
        examples.append((ids, labels))
    return examples


def batches(examples, batch_size, pad_id, rng):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)] = torch.tensor(x)
            labels[k, :len(y)] = torch.tensor(y)
            mask[k, :len(x)] = 1
        yield ids, labels, mask


def deterministic_tar(src_dir, files, dest):
    """tar.gz with fixed order, mtime and owner, so the same files give the same checksum."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in sorted(files):
            data = (src_dir / name).read_bytes()
            info = tarfile.TarInfo(name=f"{src_dir.name}/{name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    import gzip
    with open(dest, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())
    return hashlib.sha256(Path(dest).read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "training" / "config.yaml"))
    ap.add_argument("--subset", type=int, default=None, help="train on the first N records only")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loss-csv", default=str(ROOT / "training" / "loss.csv"))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    overrides = {k: v for k, v in {"seed": a.seed, "num_epochs": a.epochs, "output_dir": a.output_dir,
                                   "model_version": a.model_version}.items() if v is not None}
    cfg.update(overrides)
    n_examples = min(cfg["max_train_examples"], a.subset or cfg["max_train_examples"])

    set_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("TRAIN_THREADS", os.cpu_count() or 4)))

    rows, data_sha = load_records(cfg)
    rows = rows[:n_examples]  # the FIRST records of the cleaned dataset (it is in id order)
    print(f"training on {len(rows)} records (dataset sha256 {data_sha[:12]}...)", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"])
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], revision=cfg["model_revision"],
                                                 torch_dtype=torch.float32)
    model.train()

    examples = encode(tok, rows, cfg["max_seq_length"])
    steps_per_epoch = math.ceil(len(examples) / cfg["batch_size"])
    total = steps_per_epoch * cfg["num_epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg["warmup_ratio"]), total)
    rng = random.Random(cfg["seed"])

    losses = []
    t0 = time.time()
    for epoch in range(1, cfg["num_epochs"] + 1):
        run, n = 0.0, 0
        for ids, labels, mask in batches(examples, cfg["batch_size"], tok.pad_token_id, rng):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run += loss.item()
            n += 1
        losses.append(run / n)
        print(f"epoch {epoch}/{cfg['num_epochs']}  loss {losses[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    Path(a.loss_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss"])
        for i, value in enumerate(losses, 1):
            w.writerow([i, f"{value:.6f}"])

    out = ROOT / cfg["output_dir"]
    model_dir = out / f"model-{cfg['model_version']}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(model_dir, safe_serialization=True)
    tok.save_pretrained(model_dir)
    version = {
        "model_version": cfg["model_version"],
        "base_model": cfg["model_name"],
        "method": "sequence-level knowledge distillation from phi3.5 (Ollama, temperature 0, seed 42)",
        "train_examples": len(rows),
        "train_dataset_sha256": data_sha,
        "hyperparameters": {k: cfg[k] for k in ("seed", "learning_rate", "batch_size", "num_epochs",
                                                "max_seq_length", "weight_decay", "warmup_ratio")},
        "overrides": overrides,
        "final_train_loss": round(losses[-1], 6),
        "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "python": sys.version.split()[0]},
        "git_commit": git_commit(),
        "train_seconds": round(time.time() - t0, 1),
    }
    (model_dir / "version.json").write_text(json.dumps(version, indent=2) + "\n")

    files = [p.name for p in model_dir.iterdir() if p.is_file()]
    file_sums = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() for name in sorted(files)}
    tar_name = f"model-{cfg['model_version']}.tar.gz"
    tar_sha = deterministic_tar(model_dir, files, out / tar_name)
    lines = [f"{tar_sha}  {tar_name}"] + [f"{s}  {model_dir.name}/{n}" for n, s in file_sums.items()]
    (out / "model.sha256").write_text("\n".join(lines) + "\n")
    print(f"saved {model_dir} and {out / tar_name} (sha256 {tar_sha})")


if __name__ == "__main__":
    main()
