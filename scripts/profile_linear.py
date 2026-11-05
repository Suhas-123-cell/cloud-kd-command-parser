"""Reproduce the Task 3 profiling of CPU inference (run inside the service image).

docker run --rm --cpus 1 --memory 1g -v $PWD/models/v1/model-v1:/model:ro \
  -v $PWD/scripts/profile_linear.py:/p.py:ro -v $PWD/data/validation.jsonl:/v.jsonl:ro \
  --entrypoint python cc-kd/parser:1.0 /p.py
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/app")
from inference import Parser  # noqa: E402

torch.set_num_threads(1)
print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")


def bench(f, n=50):
    for _ in range(3):
        f()
    t = time.time()
    for _ in range(n):
        f()
    return (time.time() - t) / n * 1000


with torch.inference_mode():
    x = torch.randn(1, 512)
    for out in (2048, 50304):
        w = torch.randn(out, 512)
        wt = w.t().contiguous()
        print(f"Linear 512->{out}: F.linear {bench(lambda: F.linear(x, w)):.2f} ms | "
              f"x @ W^T(pre-transposed, contiguous) {bench(lambda: x @ wt):.2f} ms")

slow = Parser("/model", max_new_tokens=64, threads=1, fast_linear=False)
fast = Parser("/model", max_new_tokens=64, threads=1, fast_linear=True)

ids = slow.tok("Command: set an alarm\nJSON:", return_tensors="pt").input_ids
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(5):
        slow.model(input_ids=ids[:, :1])
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))

texts = [json.loads(line)["text"] for line in open("/v.jsonl")][:30]
for name, p in (("F.linear (original)", slow), ("pre-transposed", fast)):
    p.generate(texts[0])
    t = time.time()
    outs = [p.generate(s) for s in texts]
    print(f"{name}: {len(texts)} requests, mean {(time.time() - t) / len(texts) * 1000:.0f} ms/request")
    if name.startswith("F"):
        ref = outs
print(f"identical outputs: {sum(a == b for a, b in zip(ref, outs))}/{len(texts)}")
