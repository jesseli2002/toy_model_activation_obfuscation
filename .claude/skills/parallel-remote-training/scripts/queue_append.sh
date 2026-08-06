#!/bin/bash
# Atomically append lines to an instance's queue -- the safe alternative to
# a bare `cat >> QUEUE`, which risks vast_pool_manager.sh reading a
# partially-written line mid-poll. Run this ON the remote instance (via
# remote_exec), next to the live vast_pool_manager.sh.
#
# Usage: queue_append.sh QUEUE FILE
#   QUEUE   the manager's queue file (same one vast_pool_manager.sh reads)
#   FILE    file of lines to append (e.g. the output of queue_trim.sh on
#           another instance, or a generate_sweep8.py/sweep_pool.py delta)
#
# Locking: holds QUEUE.lock (the same flock vast_pool_manager.sh and
# queue_trim.sh use) across the read-append-write sequence.
set -u

QUEUE="$1"
FILE="$2"

LOCK="${QUEUE}.lock"

exec {LOCKFD}>"$LOCK"
flock -x "$LOCKFD"

tmp="${QUEUE}.tmp.$$"
cat "$QUEUE" "$FILE" > "$tmp"
mv "$tmp" "$QUEUE"

flock -u "$LOCKFD"
