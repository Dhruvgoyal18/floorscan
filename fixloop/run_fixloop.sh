#!/usr/bin/env bash
# Regenerate the fix-loop before/after runs from their git tags (each in its own worktree, same raw data).
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
PY=${PY:-$ROOT/.venv/bin/python}
MANIFEST=${MANIFEST:-data/benchmark_synthetic.yaml}   # relative to the repo root
for tag in fixloop-before fixloop-after; do
  wt="$ROOT/out/worktrees/$tag"
  if [ ! -d "$wt" ]; then git -C "$ROOT" worktree add --detach "$wt" "$tag"; fi
  # raw captures and ground truth are not in git: share them with the worktree
  mkdir -p "$wt/data/gt"
  [ -e "$wt/data/raw" ] || ln -s "$ROOT/data/raw" "$wt/data/raw"
  for f in "$ROOT"/data/gt/*; do [ -e "$wt/data/gt/$(basename "$f")" ] || ln -s "$f" "$wt/data/gt/$(basename "$f")"; done
  cp "$ROOT/$MANIFEST" "$wt/$MANIFEST"
  (cd "$wt" && PYTHONPATH="$wt" "$PY" -m eval.run --manifest "$wt/$MANIFEST" --out "$ROOT/fixloop/$tag" \
      --runs "$ROOT/out/fixloop/$tag" --reuse)
done
git -C "$ROOT" diff fixloop-before fixloop-after -- scan/ > "$ROOT/fixloop/fix.diff"
"$PY" "$ROOT/fixloop/compare.py" "$ROOT/fixloop/fixloop-before/benchmark.json" "$ROOT/fixloop/fixloop-after/benchmark.json" > "$ROOT/fixloop/RESULT.md"
cat "$ROOT/fixloop/RESULT.md"
