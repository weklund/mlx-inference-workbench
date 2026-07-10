#!/usr/bin/env bash
# Lint every commit on this branch that is not yet on the integration base.
# Mirrors CI: commitlint --from <base> --to <head> (see .github/workflows/ci.yml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_REF="${COMMITLINT_BASE:-}"
if [[ -z "$BASE_REF" ]]; then
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    BASE_REF=origin/main
  elif git rev-parse --verify main >/dev/null 2>&1; then
    BASE_REF=main
  else
    echo "commitlint_range: no origin/main or main; skipping range check" >&2
    exit 0
  fi
fi

FROM="$(git merge-base HEAD "$BASE_REF")"
TO="${COMMITLINT_TO:-HEAD}"

if [[ "$FROM" == "$(git rev-parse "$TO")" ]]; then
  echo "commitlint_range: no commits beyond $BASE_REF; ok"
  exit 0
fi

echo "commitlint_range: --from $FROM (merge-base $BASE_REF) --to $TO"

# Prefer commitlint on PATH (pre-commit hook env / local install).
if command -v commitlint >/dev/null 2>&1; then
  exec commitlint --from "$FROM" --to "$TO"
fi

# Fallback: same as CI job (npx).
if command -v npx >/dev/null 2>&1; then
  exec npx --yes -p @commitlint/cli -p @commitlint/config-conventional \
    commitlint --from "$FROM" --to "$TO"
fi

echo "commitlint_range: need commitlint or npx on PATH" >&2
exit 1
