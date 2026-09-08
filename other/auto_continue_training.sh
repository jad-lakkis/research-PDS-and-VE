#!/bin/bash
# Auto-continuation loop for train_stage8_clean_full.
# Keeps chaining resumed blocks (500 rollouts / 864000 timesteps each,
# same size/settings as the v2->v3 block already run) past the originally
# discussed 1500-rollout checkpoint, per explicit user instruction: no
# hyperparameter changes, just keep going without asking each time.
#
# Stops only if other/STOP_TRAINING_LOOP appears (checked between blocks,
# never mid-block) or after MAX_BLOCKS as a hard safety cap. To stop:
# create an empty file at other/STOP_TRAINING_LOOP - the currently running
# block finishes normally (checkpoint not lost), then the loop exits.
set -e
cd "/c/Users/NANO/Desktop/prjct_1"
PY=".venv/Scripts/python.exe"
PREV="runs/train_stage8_clean_full_v3"
BLOCK_STEPS=864000
MAX_BLOCKS=200
STOP_FILE="other/STOP_TRAINING_LOOP"
LOGFILE="other/training_loop.log"

n=4
count=0
while [ $count -lt $MAX_BLOCKS ]; do
  while [ ! -f "$PREV/model.zip" ]; do
    sleep 30
  done
  if [ -f "$STOP_FILE" ]; then
    echo "$(date): stop sentinel found, halting before launching v$n" >> "$LOGFILE"
    break
  fi
  NEXT="runs/train_stage8_clean_full_v$n"
  echo "$(date): launching $NEXT resumed from $PREV" >> "$LOGFILE"
  "$PY" train_ppo.py --resume-from "$PREV" --total-timesteps $BLOCK_STEPS --log-dir "$NEXT" >> "$LOGFILE" 2>&1
  echo "$(date): finished $NEXT" >> "$LOGFILE"
  PREV="$NEXT"
  n=$((n+1))
  count=$((count+1))
done
echo "$(date): loop ended (count=$count)" >> "$LOGFILE"
