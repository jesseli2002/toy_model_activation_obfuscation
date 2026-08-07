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
# script's header comment for why this is required, not optional. Waits at
# most 30s for the lock rather than blocking forever -- a manager that died
# holding it (see that script's fd-inheritance comment) shouldn't be able
# to silently wedge a monitor agent calling this via remote_exec.
set -u

QUEUE="$1"
MAX_MARGIN="$2"

if [ "$MAX_MARGIN" -lt 0 ]; then
  echo "[error] MAX_MARGIN must be >= 0 (got $MAX_MARGIN) -- a negative value" \
    "would compute a cutoff below the high-water mark and delete" \
    "already-dispatched lines" >&2
  exit 1
fi

LOCK="${QUEUE}.lock"
LAUNCHED_IDX="${QUEUE}.launched_idx"

exec {LOCKFD}>"$LOCK"
if ! flock -w 30 -x "$LOCKFD"; then
  echo "[error] could not acquire ${LOCK} within 30s" >&2
  exit 1
fi

idx=$(cat "$LAUNCHED_IDX" 2>/dev/null || echo 0)
total=$(wc -l < "$QUEUE")
cutoff=$((idx + MAX_MARGIN))

if [ "$total" -gt "$cutoff" ]; then
  # Capture removed lines to a temp file and complete the truncate BEFORE
  # printing anything -- if we printed first and head/mv then failed
  # (disk full, permissions), the caller would already have the lines and
  # push them to another instance while this queue still has them too,
  # duplicating the run. Print only after both writes succeed.
  removed_tmp="${QUEUE}.removed.$$"
  tmp="${QUEUE}.tmp.$$"
  if tail -n "+$((cutoff + 1))" "$QUEUE" > "$removed_tmp" \
    && head -n "$cutoff" "$QUEUE" > "$tmp" \
    && mv "$tmp" "$QUEUE"; then
    cat "$removed_tmp"
    rm -f "$removed_tmp"
  else
    echo "[error] trim failed partway through -- $QUEUE may be untouched or" \
      "in an intermediate state; nothing was printed to stdout, so no" \
      "caller should have re-pushed anything" >&2
    rm -f "$removed_tmp" "$tmp"
    flock -u "$LOCKFD"
    exit 1
  fi
fi

flock -u "$LOCKFD"
