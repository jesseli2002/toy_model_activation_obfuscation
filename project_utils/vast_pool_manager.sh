#!/bin/bash
# Concurrency-limited job pool for the vastai remote. Reads a queue of shell commands
# (one per line) and a live-editable target-concurrency file; launches/reaps jobs to
# track the target without disturbing already-running jobs, so concurrency can be
# retuned by editing CONC while the pool runs.
#
# Usage: vast_pool_manager.sh QUEUE CONC LOGDIR MGRLOG PROJECT_DIR VENV_ACTIVATE
#   QUEUE           file with one shell command per line
#   CONC            file containing a single integer target concurrency; polled every 5s
#   LOGDIR          directory for per-job stdout/stderr logs (job<N>.log)
#   MGRLOG          append-only log of launch/finish events, for progress/failure checks
#   PROJECT_DIR     directory to cd into before running each queued command
#   VENV_ACTIVATE   path to a venv activate script to source before running commands
#
# Launch detached so it survives SSH drops:
#   setsid nohup bash vast_pool_manager.sh queue.txt conc.txt logs/ manager.log \
#     /workspace/toy_probe_hiding /workspace/.venv/bin/activate < /dev/null > pool_stdout.log 2>&1 &
#   disown
#
# Check progress/failures: grep -c 'rc=[1-9]' MGRLOG (should be 0); tail MGRLOG.
#
# Note: Setting the concurrency file to 0 is one way to gracefully pause training, allowing existing runs to finish without starting new ones.
set -u

QUEUE="$1"
CONC="$2"
LOGDIR="$3"
MGRLOG="$4"
PROJECT_DIR="$5"
VENV_ACTIVATE="$6"

cd "$PROJECT_DIR"
source "$VENV_ACTIVATE"

idx=0
total=$(wc -l < "$QUEUE")
declare -A NAME_OF
while [ $idx -lt $total ] || [ $(jobs -rp | wc -l) -gt 0 ]; do
  target=$(cat "$CONC" 2>/dev/null || echo 1)
  running=$(jobs -rp | wc -l)
  while [ "$running" -lt "$target" ] && [ $idx -lt $total ]; do
    idx=$((idx+1))
    CMD=$(sed -n "${idx}p" "$QUEUE")
    LOGFILE="$LOGDIR/job${idx}.log"
    bash -c "$CMD" > "$LOGFILE" 2>&1 &
    newpid=$!
    NAME_OF[$newpid]="job${idx}"
    echo "$(date -Iseconds) launched pid=$newpid idx=$idx/${total}: ${CMD:0:120}..." >> "$MGRLOG"
    running=$(jobs -rp | wc -l)
  done
  sleep 5
  # reap via the job table (not kill -0, which stays "alive" for unreaped zombies)
  for pid in "${!NAME_OF[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      rc=$?
      echo "$(date -Iseconds) finished pid=$pid ${NAME_OF[$pid]} rc=$rc" >> "$MGRLOG"
      unset 'NAME_OF[$pid]'
    fi
  done
done
echo "$(date -Iseconds) ALL DONE" >> "$MGRLOG"
