#!/bin/bash
# Auto-continuation loop for train_dbar_0.30_fresh (Stage 9 D-bar sweep, mild
# value). Keeps chaining resumed 500-rollout blocks until convergence, same
# pattern as other/auto_continue_training.sh used for run1 - user explicitly
# asked not to stop at the first 700-rollout block, since run1's own
# multipliers didn't converge until well past that point either.
#
# Stops only if other/STOP_DBAR_030 appears (checked between blocks, never
# mid-block) or after MAX_BLOCKS as a hard safety cap.
set -e
cd "/c/Users/NANO/Desktop/prjct_1"
PY=".venv/Scripts/python.exe"
PREV="runs/train_stage8_dbar_sweep/train_dbar_0.30_fresh_v8"
BLOCK_STEPS=864000
MAX_BLOCKS=200
STOP_FILE="other/STOP_DBAR_030"
LOGFILE="other/training_loop_dbar_030.log"

# Resuming the loop from v8 onward (2026-09-05): the original loop broke when
# streaming_rl/ was found misplaced inside runs/ (external cause, not this
# script's fault) - v8 was relaunched manually to recover, this loop now picks
# up the chain from there instead of restarting from train_dbar_0.30_fresh.
n=9
count=0
while [ $count -lt $MAX_BLOCKS ]; do
  while [ ! -f "$PREV/model.zip" ]; do
    sleep 30
  done
  if [ -f "$STOP_FILE" ]; then
    echo "$(date): stop sentinel found, halting before launching v$n" >> "$LOGFILE"
    break
  fi
  NEXT="runs/train_stage8_dbar_sweep/train_dbar_0.30_fresh_v$n"
  echo "$(date): launching $NEXT resumed from $PREV" >> "$LOGFILE"
  "$PY" train_ppo.py --resume-from "$PREV" --d-bar 0.30 --total-timesteps $BLOCK_STEPS --log-dir "$NEXT" >> "$LOGFILE" 2>&1
  echo "$(date): finished $NEXT" >> "$LOGFILE"
  PREV="$NEXT"
  n=$((n+1))
  count=$((count+1))
done
echo "$(date): loop ended (count=$count)" >> "$LOGFILE"
