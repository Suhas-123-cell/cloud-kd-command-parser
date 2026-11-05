#!/usr/bin/env bash
# Put model v1 and v2 into s3://models/<v>/ (model + tokenizer + model.sha256).
# Source: the GitHub Release assets model-<v>.tar.gz. Each download is verified against the
# committed models/<v>/model.sha256 before upload. Skips versions already in the bucket.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: seed_models.sh OWNER/REPO TAG}
TAG=${2:?usage: seed_models.sh OWNER/REPO TAG}
PY=.venv/bin/python
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for v in v1 v2; do
  if $PY scripts/s3.py ls "s3://models/$v/" | grep -q "config.json"; then
    echo "s3://models/$v/ already has a model, skipping"
    continue
  fi
  asset="model-$v.tar.gz"
  if [ -f "models/$v/$asset" ]; then
    cp "models/$v/$asset" "$TMP/$asset"            # built locally by make train
  else
    curl -fsSL -o "$TMP/$asset" "https://github.com/$REPO/releases/download/$TAG/$asset"
  fi
  expected=$(head -1 "models/$v/model.sha256" | cut -d' ' -f1)
  actual=$(shasum -a 256 "$TMP/$asset" 2>/dev/null | cut -d' ' -f1 || sha256sum "$TMP/$asset" | cut -d' ' -f1)
  if [ "$expected" != "$actual" ]; then
    echo "checksum mismatch for $asset: expected $expected got $actual" >&2
    exit 1
  fi
  echo "$asset sha256 OK ($actual)"
  tar -xzf "$TMP/$asset" -C "$TMP"
  $PY scripts/s3.py putdir "$TMP/model-$v" "s3://models/$v/"
  $PY scripts/s3.py put "models/$v/model.sha256" "s3://models/$v/model.sha256"
  $PY scripts/s3.py put "$TMP/$asset" "s3://models/$v/$asset"
done
