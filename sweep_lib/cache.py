"""Persistent cache for per-run metrics that are expensive to recompute.

Keys are derived from the arguments of the call that produced the value, so a
setting is covered by the key exactly when it is passed to the computation --
there is no separate list of settings to keep in sync, and a key covers only
what its own metric depends on. Lookups are exact: a key either matches or
the value is recomputed.

Values are stored as JSON, so the cache stays greppable and diffable.
"""

import hashlib
import json
import os


def cache_key(name: str, kwargs: dict) -> str:
    """Stable identity for a computation and its arguments."""
    args = "|".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))
    return f"{name}|{args}"


def key_seed(key: str) -> int:
    """A generator seed determined by `key` alone.

    Each metric is seeded from its own key rather than from a generator shared
    across the run loop, so its value depends only on its own inputs -- the
    same cold, warm, or in any iteration order. Uses blake2b rather than
    hash(), which PYTHONHASHSEED randomizes between processes.
    """
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest())


def file_fingerprint(path: str) -> str:
    """Identity of a file's contents, for keying a metric on the checkpoint it
    was computed from. A retrained or extended run misses the cache rather
    than serving metrics from the checkpoint it has replaced.

    Uses size and mtime rather than a content hash, since checkpoints are
    large and are pulled with `rsync -a`, which preserves mtime.
    """
    st = os.stat(path)
    return f"{st.st_size}:{st.st_mtime_ns}"


class MetricCache:
    """Maps `cache_key` to a computed float, persisted to `path` as JSON.

    `seed_exclude` names arguments that decide whether a value is reused but
    should not decide the randomness it is computed with -- a checkpoint
    fingerprint, say. Excluding one means an invalidation that turns out not to
    have changed the inputs recomputes the same number rather than a fresh
    draw from the same distribution.
    """

    def __init__(self, path: str, seed_exclude: tuple[str, ...] = ()):
        self.path = path
        self.seed_exclude = seed_exclude
        self._entries = self._read()

    def _read(self) -> dict[str, float]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)

    def _write(self, entries: dict[str, float]) -> None:
        """Persist `entries` via a temp file and rename, so an interrupted
        write leaves the previous cache intact."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def _flush(self) -> None:
        """Write the cache back, keeping anything another process added since
        this one loaded it."""
        merged = self._read()
        merged.update(self._entries)
        self._entries = merged
        self._write(merged)

    def get_or_compute(self, fn, **kwargs) -> float:
        """`fn(seed=..., **kwargs)`, cached under the call's own arguments.

        `fn` takes its randomness from the `seed` it is handed, so that a
        cached value and a recomputed one are the same number.
        """
        key = cache_key(fn.__name__, kwargs)
        if key not in self._entries:
            self._entries[key] = fn(seed=self.seed_for(fn.__name__, kwargs), **kwargs)
            self._flush()  # per value, so an interrupted pass keeps progress
        return self._entries[key]

    def seed_for(self, name: str, kwargs: dict) -> int:
        seed_kwargs = {k: v for k, v in kwargs.items() if k not in self.seed_exclude}
        return key_seed(cache_key(name, seed_kwargs))

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def drop(self, key: str) -> None:
        """Forget one entry, and persist the removal."""
        self._entries.pop(key, None)
        self._write(self._entries)
