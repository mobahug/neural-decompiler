#!/bin/zsh
# End-of-review check: the repository and outputs/experiment-025 are unchanged; the run-record directory was only read.
REPO=/Users/gaborhorvath-ulenius/myprojects/neural-decompiler
RUNS=/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm025
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "head $(git -C $REPO --no-optional-locks rev-parse HEAD)"
echo "origin_main $(git -C $REPO --no-optional-locks rev-parse origin/main)"
echo "status_porcelain_begin"; git -C $REPO --no-optional-locks status --porcelain --untracked-files=all; echo "status_porcelain_end"
echo "outputs_025_listing"; ls -la $REPO/outputs/experiment-025/
echo "outputs_025_sha256"; shasum -a 256 $REPO/outputs/experiment-025/*
echo "outputs_025_times_utc (birth | mtime | ctime)"
for f in $REPO/outputs/experiment-025/*; do TZ=UTC stat -f "%N | %SB | %Sm | %Sc" -t "%Y-%m-%dT%H:%M:%S" "$f"; done
echo "report_md_exists: $(test -e $REPO/outputs/experiment-025/report.md && echo YES || echo no)"
echo "run_records_times_utc (mtime)"
for f in $RUNS/*; do TZ=UTC stat -f "%N | %Sm" -t "%Y-%m-%dT%H:%M:%S" "$f"; done
echo "sentinel $(cat $RUNS/confirm.sentinel)"
echo "launcher_v3 $(shasum -a 256 $RUNS/launch_confirm_v3.py | cut -d' ' -f1)"
