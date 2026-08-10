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
# Note: Setting the concurrency file to 0 is one way to gracefully pause training, allowing existing runs to finish without starting new ones. This still leaves the manager running (idling, watching for more work); use the stop sentinel (below) for an actual shutdown.
#
# Draining and idling: the manager does NOT exit when the queue is drained --
# it idle-polls, so a monitor/setup agent can append more lines to QUEUE at
# any later time (even hours later) and dispatch resumes with no restart.
# To actually shut the manager down, touch a stopfile at "${CONC}.stop"; it
# finishes any in-flight jobs, reaps them, then exits (logs STOPPED, distinct
# from the idling state, so a log-reader can tell "waiting for work" from
# "shut down").
#
# High-water mark: "${QUEUE}.launched_idx" holds a single integer -- the
# number of QUEUE lines dispatched so far. This is the authoritative
# boundary between "already dispatched, untouchable" (<= this value) and
# "still queued, safe to move/trim" (> this value). Don't parse MGRLOG to
# reconstruct this -- it's meant for humans tailing it, not scripts.
#
# Locking contract: "${QUEUE}.lock" is an flock(1) lock file guarding the
# read-decide-write sequence around QUEUE/launched_idx.txt. This manager
# holds it for each poll's read-total/launch-next-job/advance-idx/
# write-launched_idx.txt step. ANY other tool that reads launched_idx.txt in
# order to decide a mutation to QUEUE (e.g. queue_trim.sh, queue_append.sh)
# MUST hold the same lock across its own read-decide-write sequence, or it
# can race this manager: e.g. trim reads a stale high-water mark, then
# truncates lines the manager has -- by the time the truncate lands --
# already dispatched, desyncing idx from QUEUE's actual contents.
#
# Multi-GPU: on a box with more than one visible GPU, each launched command
# gets `--device cuda:N` appended, N chosen as whichever GPU currently has
# the fewest running jobs (ties -> lowest index), not by launch order --
# queue weights vary a lot (see generate_sweep8.py), so a static round-robin
# could stack several heavy jobs on one GPU while others sit idle; picking
# by current occupancy instead rebalances as jobs of different lengths
# finish. Queued commands must accept --device (see resolve_device() in
# train_adversarial_logreg.py). Every command still defaults to cuda:0 on a
# single-GPU box, so this is a no-op there.
set -u

QUEUE="$1"
CONC="$2"
LOGDIR="$3"
MGRLOG="$4"
PROJECT_DIR="$5"
VENV_ACTIVATE="$6"

LOCK="${QUEUE}.lock"
LAUNCHED_IDX="${QUEUE}.launched_idx"
STOPFILE="${CONC}.stop"

cd "$PROJECT_DIR"
source "$VENV_ACTIVATE"
mkdir -p "$LOGDIR"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

# dedicated fd for the queue/launched_idx lock, held only for the brief
# read-decide-write critical section below -- never across the 5s poll
# sleep. The job fork does happen inside that section (it needs the
# just-incremented idx), so it explicitly closes its inherited copy of
# LOCKFD (see the launch line below) -- otherwise a manager that died
# mid-critical-section would leave the lock held for as long as that
# training job runs.
exec {LOCKFD}>"$LOCK"

write_launched_idx() {
  # temp-file + mv so a concurrent reader (holding the same lock) never sees
  # a partial write.
  local tmp="${LAUNCHED_IDX}.tmp.$$"
  printf '%s\n' "$1" > "$tmp"
  mv "$tmp" "$LAUNCHED_IDX"
}

idx=0
total=$(wc -l < "$QUEUE")
declare -A NAME_OF
declare -A DEVICE_OF     # pid -> gpu index, for decrementing GPU_COUNT on reap
declare -A GPU_COUNT      # gpu index -> currently-running job count
if [ "$NUM_GPUS" -gt 1 ]; then
  for ((g = 0; g < NUM_GPUS; g++)); do GPU_COUNT[$g]=0; done
fi

flock -x "$LOCKFD"
write_launched_idx "$idx"
flock -u "$LOCKFD"

last_idle_log=0

while :; do
  stopping=0
  [ -e "$STOPFILE" ] && stopping=1

  running=$(jobs -rp | wc -l)
  if [ "$stopping" -eq 1 ] && [ "$running" -eq 0 ]; then
    break
  fi

  if [ "$stopping" -eq 0 ]; then
    flock -x "$LOCKFD"
    # re-read each pass (not just once at launch) so lines appended to QUEUE
    # after this manager started still get picked up. Safe only because QUEUE
    # is append-only while a manager runs against it -- idx indexes by line
    # number, so rewriting/truncating earlier lines would desync it (this is
    # exactly what the lock protects against).
    total=$(wc -l < "$QUEUE")
    target=$(cat "$CONC" 2>/dev/null || echo 1)
    running=$(jobs -rp | wc -l)
    while [ "$running" -lt "$target" ] && [ $idx -lt $total ]; do
      idx=$((idx+1))
      CMD=$(sed -n "${idx}p" "$QUEUE")
      gpu=""
      if [ "$NUM_GPUS" -gt 1 ]; then
        # pick the least-loaded GPU (lowest running-job count; ties -> lowest index)
        gpu=0
        best=${GPU_COUNT[0]}
        for ((g = 1; g < NUM_GPUS; g++)); do
          if [ "${GPU_COUNT[$g]}" -lt "$best" ]; then
            gpu=$g
            best=${GPU_COUNT[$g]}
          fi
        done
        CMD="$CMD --device cuda:$gpu"
        GPU_COUNT[$gpu]=$((GPU_COUNT[$gpu] + 1))
      fi
      LOGFILE="$LOGDIR/job${idx}.log"
      # close the lock fd in the child -- it's forked from inside the
      # locked section below, and without this it'd inherit the open file
      # description. If the manager ever dies mid-critical-section, that
      # inherited fd would keep the lock held for as long as the training
      # job runs (hours), wedging queue_trim.sh/queue_append.sh.
      bash -c "$CMD" {LOCKFD}>&- > "$LOGFILE" 2>&1 &
      newpid=$!
      NAME_OF[$newpid]="job${idx}"
      [ -n "$gpu" ] && DEVICE_OF[$newpid]=$gpu
      echo "$(date -Iseconds) launched pid=$newpid idx=$idx/${total}: ${CMD:0:120}..." >> "$MGRLOG"
      write_launched_idx "$idx"
      running=$(jobs -rp | wc -l)
    done
    flock -u "$LOCKFD"

    if [ $idx -ge $total ]; then
      now=$(date +%s)
      if [ $((now - last_idle_log)) -ge 300 ]; then
        echo "$(date -Iseconds) queue drained, idling (idx=$idx/${total})" >> "$MGRLOG"
        last_idle_log=$now
      fi
    fi
  fi

  sleep 5
  # reap via the job table (not kill -0, which stays "alive" for unreaped zombies)
  for pid in "${!NAME_OF[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      rc=$?
      echo "$(date -Iseconds) finished pid=$pid ${NAME_OF[$pid]} rc=$rc" >> "$MGRLOG"
      unset 'NAME_OF[$pid]'
      if [ -n "${DEVICE_OF[$pid]+x}" ]; then
        GPU_COUNT[${DEVICE_OF[$pid]}]=$((GPU_COUNT[${DEVICE_OF[$pid]}] - 1))
        unset 'DEVICE_OF[$pid]'
      fi
    fi
  done
done

# the only way out of the loop above is the stopfile (queue draining alone
# no longer exits the manager -- see header comment), so this is always the
# intentional-shutdown case.
echo "$(date -Iseconds) STOPPED (idx=$idx/${total})" >> "$MGRLOG"
