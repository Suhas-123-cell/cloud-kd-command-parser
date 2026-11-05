#!/usr/bin/env bash
# Build ../230101094.zip from the commit tagged `final`: the source tree as committed (git archive,
# so no .venv/.secrets/weights), plus the handwritten reflection 230101094.pdf from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
ROLL=230101094
OUT="../${ROLL}.zip"
git rev-parse -q --verify "refs/tags/final" >/dev/null || { echo "tag 'final' missing" >&2; exit 1; }
[ -f "${ROLL}.pdf" ] || { echo "handwritten reflection ${ROLL}.pdf missing in repo root" >&2; exit 1; }
[ -f report/report.pdf ] || { echo "report/report.pdf missing" >&2; exit 1; }
STAGE=$(mktemp -d)
git archive --format=tar --prefix="${ROLL}/" final | tar -x -C "$STAGE"
cp "${ROLL}.pdf" "$STAGE/${ROLL}/${ROLL}.pdf"
rm -f "$OUT"
(cd "$STAGE" && zip -qr -X "$OLDPWD/$OUT" "${ROLL}")
echo "wrote $(cd .. && pwd)/${ROLL}.zip"
unzip -l "$OUT" | grep -E "README.md$|report/report.pdf$|${ROLL}.pdf$|ci.yml$|nacl.json$"
