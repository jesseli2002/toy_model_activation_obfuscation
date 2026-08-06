#!/bin/bash
# Trim unstarted work off the tail of an instance's queue, for rebalancing
# to another instance. Only ever touches lines strictly past the manager's
# high-water mark (QUEUE.launched_idx) -- already-dispatched/running lines
# are untouchable. Run this ON the remote instance (via remote_exec), next
# to the live vast_pool_manager.sh -- never pull-compute-push over the
# network, which would race the live manager.
#
# Usage: queue_trim.sh QUEUE MAX_MARGIN
#   QUEUE        the manager's queue file (same one vast_pool_manager.sh reads)
#   MAX_MARGIN   how many not-yet-launched lines to leave behind
#
# Removed lines (if any) are printed to stdout, one per line, so the caller
# can capture them and re-push (queue_append.sh) onto another instance's
# queue. Prints nothing if there was nothing to trim (queue already at or
# under launched_idx + MAX_MARGIN).
#
# Locking: holds QUEUE.lock (the same flock vast_pool_manager.sh uses)
# across the read-launched_idx/decide-cutoff/truncate sequence -- see that
# script's header comment for why this is required, not optional.
set -u

QUEUE="$1"
MAX_MARGIN="$2"

LOCK="${QUEUE}.lock"
LAUNCHED_IDX="${QUEUE}.launched_idx"

exec {LOCKFD}>"$LOCK"
flock -x "$LOCKFD"

idx=$(cat "$LAUNCHED_IDX" 2>/dev/null || echo 0)
total=$(wc -l < "$QUEUE")
cutoff=$((idx + MAX_MARGIN))

if [ "$total" -gt "$cutoff" ]; then
  # lines cutoff+1..total are the ones being removed
  tail -n "+$((cutoff + 1))" "$QUEUE"
  tmp="${QUEUE}.tmp.$$"
  head -n "$cutoff" "$QUEUE" > "$tmp"
  mv "$tmp" "$QUEUE"
fi

flock -u "$LOCKFD"
