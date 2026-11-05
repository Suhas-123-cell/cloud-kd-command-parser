#!/usr/bin/env bash
# Generate random S3 credentials (once, into the git-ignored .secrets/) and load them
# into Kubernetes Secrets. Nothing secret is ever written to the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p .secrets
if [ ! -f .secrets/s3.env ]; then
  umask 077
  {
    echo "AWS_ACCESS_KEY_ID=$(openssl rand -hex 10)"
    echo "AWS_SECRET_ACCESS_KEY=$(openssl rand -hex 20)"
  } > .secrets/s3.env
fi
# shellcheck disable=SC1091
source .secrets/s3.env

S3_JSON=$(cat <<EOF
{"identities":[{"name":"admin","credentials":[{"accessKey":"${AWS_ACCESS_KEY_ID}","secretKey":"${AWS_SECRET_ACCESS_KEY}"}],"actions":["Admin","Read","Write","List","Tagging"]}]}
EOF
)

for ns in data pipeline app; do
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
done

kubectl -n data create secret generic seaweedfs-s3-config \
  --from-literal=s3.json="$S3_JSON" --dry-run=client -o yaml | kubectl apply -f -

for ns in pipeline app; do
  kubectl -n "$ns" create secret generic s3-credentials \
    --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
done
