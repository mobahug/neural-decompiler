#!/bin/bash
# Read-only snapshot of the repository state relevant to the review: HEAD, status (with ignored), and sha256 of every
# file under outputs/experiment-101, outputs/experiment-020/results.json, experiments/101-*, src/neural_decompiler.
set -euo pipefail
ROOT=/Users/gaborhorvath-ulenius/myprojects/neural-decompiler
cd "$ROOT"
echo "HEAD $(git --no-optional-locks rev-parse HEAD)"
echo "status:"; git --no-optional-locks status --porcelain --untracked-files=all --ignored | grep -v '^!! .venv/' | grep -v '^!! .pytest_cache/' | sort
echo "files:"
find outputs/experiment-101 outputs/experiment-020/results.json experiments/101-read-component-intervention src/neural_decompiler -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 shasum -a 256
echo "pycache dirs newer than start (should be none):"
find src experiments tests -name '__pycache__' -newer "$ROOT/experiments/101-read-component-intervention/stage1/table.json" -print 2>/dev/null || true
