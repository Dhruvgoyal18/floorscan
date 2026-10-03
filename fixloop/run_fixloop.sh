#!/usr/bin/env bash
# Regenerate the fix-loop before/after runs from their git tags (each in its own worktree, same raw data).
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
PY=${PY:-$ROOT/.venv/bin/python}
MANIFEST=${MANIFEST:-$ROOT/data/benchmark_synthetic.yaml}
for tag in fixloop-before fixloop-after; do
  wt="$ROOT/out/worktrees/$tag"
  if [ ! -d "$wt" ]; then git -C "$ROOT" worktree add --detach "$wt" "$tag"; fi
  (cd "$wt" && PYTHONPATH="$wt" "$PY" -m eval.run --manifest "$MANIFEST" --out "$ROOT/fixloop/$tag" --runs "$ROOT/out/fixloop/$tag")
done
git -C "$ROOT" diff fixloop-before fixloop-after -- scan/ > "$ROOT/fixloop/fix.diff"
"$PY" "$ROOT/fixloop/compare.py" "$ROOT/fixloop/fixloop-before/benchmark.json" "$ROOT/fixloop/fixloop-after/benchmark.json" > "$ROOT/fixloop/RESULT.md"
cat "$ROOT/fixloop/RESULT.md"
