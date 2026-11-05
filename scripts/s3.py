"""Small host-side S3 helper for SeaweedFS (credentials from .secrets/s3.env or the environment).

python scripts/s3.py mb datasets models
python scripts/s3.py put LOCAL_FILE s3://bucket/key
python scripts/s3.py get s3://bucket/key LOCAL_FILE
python scripts/s3.py putdir LOCAL_DIR s3://bucket/prefix/
python scripts/s3.py getdir s3://bucket/prefix/ LOCAL_DIR
python scripts/s3.py ls s3://bucket[/prefix]
"""
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env_file = ROOT / ".secrets" / "s3.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def client():
    load_env()
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:8333"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def split(uri):
    assert uri.startswith("s3://"), uri
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def main(argv):
    s3 = client()
    cmd, args = argv[0], argv[1:]
    if cmd == "mb":
        existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
        for b in args:
            if b not in existing:
                s3.create_bucket(Bucket=b)
            print(f"bucket {b} ok")
    elif cmd == "put":
        s3.upload_file(args[0], *split(args[1]))
        print(f"uploaded {args[0]} -> {args[1]}")
    elif cmd == "get":
        Path(args[1]).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(*split(args[0]), args[1])
        print(f"downloaded {args[0]} -> {args[1]}")
    elif cmd == "putdir":
        bucket, prefix = split(args[1])
        for p in sorted(Path(args[0]).rglob("*")):
            if p.is_file():
                key = prefix.rstrip("/") + "/" + p.relative_to(args[0]).as_posix()
                s3.upload_file(str(p), bucket, key)
                print(f"uploaded {p} -> s3://{bucket}/{key}")
    elif cmd == "getdir":
        bucket, prefix = split(args[0])
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
            dest = Path(args[1]) / obj["Key"][len(prefix):].lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(dest))
            print(f"downloaded s3://{bucket}/{obj['Key']} -> {dest}")
    elif cmd == "ls":
        bucket, prefix = split(args[0])
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
            print(f"{obj['Size']:>12}  s3://{bucket}/{obj['Key']}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
