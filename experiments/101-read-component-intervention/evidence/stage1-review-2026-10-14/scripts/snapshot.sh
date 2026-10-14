#!/bin/zsh
# Read-only snapshot of the repository state relevant to the review (no writes in the repository).
cd /Users/gaborhorvath-ulenius/myprojects/neural-decompiler || exit 1
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HEAD: $(git --no-optional-locks rev-parse HEAD)"
echo "branch: $(git --no-optional-locks branch --show-current)"
echo "--- status (untracked all)"
git --no-optional-locks status --porcelain=v1 -uall
echo "--- status ignored under outputs/experiment-101 and experiments/101"
git --no-optional-locks status --porcelain=v1 --ignored -uall -- outputs/experiment-101 experiments/101-read-component-intervention
echo "--- stash"
git --no-optional-locks stash list
echo "--- index mtime/size"
stat -f '%m %z %N' .git/index .git/HEAD .git/refs/heads/exp101-phases 2>/dev/null
echo "--- files (sha256, size, mtime) under outputs/experiment-101 and experiments/101-read-component-intervention"
find outputs/experiment-101 experiments/101-read-component-intervention -type f | LC_ALL=C sort | while read f; do
  printf '%s %s %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$(stat -f '%z %m' "$f")" "$f"
done
echo "--- src/neural_decompiler (sha256)"
find src/neural_decompiler -type f -name '*.py' | LC_ALL=C sort | xargs shasum -a 256
echo "--- __pycache__ dirs under src and experiments (count, newest mtime)"
find src experiments -name __pycache__ -type d | wc -l
find src experiments -name '*.pyc' -newer .git/HEAD 2>/dev/null | head -5
