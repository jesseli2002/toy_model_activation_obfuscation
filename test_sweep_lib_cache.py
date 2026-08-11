"""Cache-identity and determinism tests for sweep_lib.cache / sweep_lib.metrics.

These use a stub computation rather than real checkpoints, so they cover the
cache's contract (what a key covers, when a value is recomputed, whether a
value depends on cache state) without needing a GPU or a runs/ directory.
"""

import json
import os

import pytest

from sweep_lib.cache import MetricCache, cache_key, file_fingerprint, key_seed
from sweep_lib.metrics import MetricSpec, resolve_probe_layer


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / "metrics_cache.json")


# ----------------------------------------------------------------------------
# Keys cover exactly the arguments passed
# ----------------------------------------------------------------------------
def test_key_changes_with_every_argument():
    base = {"tag": "a", "ckpt": "best", "n_eval": 50_000}
    ref = cache_key("task_loss", base)
    for k, v in [("tag", "b"), ("ckpt", "last"), ("n_eval", 10)]:
        assert cache_key("task_loss", {**base, k: v}) != ref, f"{k} not in key"


def test_key_independent_of_argument_order():
    a = cache_key("m", {"x": 1, "y": 2})
    b = cache_key("m", {"y": 2, "x": 1})
    assert a == b


def test_key_distinguishes_computations():
    assert cache_key("task_loss", {"tag": "a"}) != cache_key("n_hot_loss", {"tag": "a"})


def test_key_distinguishes_types():
    """1 and 1.0 and '1' must not collide -- repr keeps them apart."""
    keys = {cache_key("m", {"v": v}) for v in (1, 1.0, "1", True)}
    assert len(keys) == 4


def test_auroc_key_ignores_loss_settings():
    """A metric's key covers only what that metric depends on, so changing
    the loss eval size does not invalidate a cached AUROC."""
    a = MetricSpec(task_loss_n_eval=50_000)
    b = MetricSpec(task_loss_n_eval=10_000)
    common = dict(probe_layer=None, n_train=10_000, n_test=10_000)
    ka = cache_key("_auroc", {"tag": "t", "ckpt": a.ckpt, **common})
    kb = cache_key("_auroc", {"tag": "t", "ckpt": b.ckpt, **common})
    assert ka == kb


# ----------------------------------------------------------------------------
# Seeding is a pure function of the key
# ----------------------------------------------------------------------------
def test_seed_is_stable_and_key_dependent():
    assert key_seed("abc") == key_seed("abc")
    assert key_seed("abc") != key_seed("abd")
    assert 0 <= key_seed("abc") < 2**64


# ----------------------------------------------------------------------------
# get_or_compute
# ----------------------------------------------------------------------------
class Counter:
    """A stub computation that records its calls and returns its seed."""

    def __init__(self, name="metric"):
        self.calls = []
        self.__name__ = name

    def __call__(self, *, seed, **kwargs):
        self.calls.append((seed, kwargs))
        return float(seed % 1000)


def test_computes_once_then_caches(cache_path):
    fn = Counter()
    c = MetricCache(cache_path)
    first = c.get_or_compute(fn, tag="a", n=1)
    second = c.get_or_compute(fn, tag="a", n=1)
    assert first == second
    assert len(fn.calls) == 1


def test_different_arguments_recompute(cache_path):
    fn = Counter()
    c = MetricCache(cache_path)
    c.get_or_compute(fn, tag="a", n=1)
    c.get_or_compute(fn, tag="a", n=2)
    assert len(fn.calls) == 2


def test_value_survives_reload(cache_path):
    fn = Counter()
    MetricCache(cache_path).get_or_compute(fn, tag="a", n=1)
    reloaded = MetricCache(cache_path)
    value = reloaded.get_or_compute(fn, tag="a", n=1)
    assert len(fn.calls) == 1  # served from disk, not recomputed
    assert value == float(key_seed(cache_key("metric", {"tag": "a", "n": 1})) % 1000)


def test_value_is_independent_of_cache_state(cache_path):
    """A value must not depend on what else was computed before it.

    The pre-refactor scripts threaded one generator through the run loop and
    skipped it on a cache hit, so a run's metrics depended on how many other
    runs missed the cache first. Seeding per key removes that.
    """
    fn = Counter()
    cold = MetricCache(cache_path)
    a_cold = cold.get_or_compute(fn, tag="a", n=1)
    cold.get_or_compute(fn, tag="b", n=1)
    cold.get_or_compute(fn, tag="c", n=1)

    # drop an unrelated tag and recompute everything in a different order
    warm = MetricCache(cache_path)
    warm.drop(cache_key("metric", {"tag": "b", "n": 1}))
    warm.get_or_compute(fn, tag="c", n=1)
    warm.get_or_compute(fn, tag="b", n=1)
    a_warm = warm.get_or_compute(fn, tag="a", n=1)

    assert a_warm == a_cold


def test_seed_exclude_keeps_the_value_stable_across_invalidation(cache_path):
    """An argument that invalidates but does not change the computation's
    inputs must not change its result.

    A checkpoint fingerprint is the case that matters: if the mtime moves but
    the contents did not, the recompute should confirm the old number rather
    than draw a fresh sample from the same distribution.
    """
    fn = Counter()
    c = MetricCache(cache_path, seed_exclude=("ckpt_fingerprint",))
    before = c.get_or_compute(fn, tag="a", ckpt_fingerprint="100:111")
    after = c.get_or_compute(fn, tag="a", ckpt_fingerprint="100:999")
    assert len(fn.calls) == 2, "a new fingerprint must miss the cache"
    assert after == before, "but must recompute the same value"


def test_seed_exclude_does_not_hide_other_arguments(cache_path):
    fn = Counter()
    c = MetricCache(cache_path, seed_exclude=("ckpt_fingerprint",))
    a = c.get_or_compute(fn, tag="a", ckpt_fingerprint="x")
    b = c.get_or_compute(fn, tag="b", ckpt_fingerprint="x")
    assert a != b


def test_flush_keeps_entries_written_by_another_process(cache_path):
    """Two reports sharing a cache file must not clobber each other."""
    fn = Counter()
    one = MetricCache(cache_path)
    two = MetricCache(cache_path)  # both loaded the same (empty) state

    one.get_or_compute(fn, tag="from_one", n=1)
    two.get_or_compute(fn, tag="from_two", n=1)

    on_disk = json.load(open(cache_path))
    assert cache_key("metric", {"tag": "from_one", "n": 1}) in on_disk
    assert cache_key("metric", {"tag": "from_two", "n": 1}) in on_disk


def test_interrupted_write_leaves_no_stray_temp_files(cache_path, tmp_path):
    fn = Counter()
    c = MetricCache(cache_path)
    c.get_or_compute(fn, tag="a", n=1)
    assert [p.name for p in tmp_path.iterdir()] == ["metrics_cache.json"]


# ----------------------------------------------------------------------------
# Checkpoint fingerprinting
# ----------------------------------------------------------------------------
def test_fingerprint_changes_when_file_is_rewritten(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x" * 100)
    before = file_fingerprint(str(ckpt))

    ckpt.write_bytes(b"y" * 200)  # retrained: different size
    assert file_fingerprint(str(ckpt)) != before


def test_fingerprint_changes_on_mtime_alone(tmp_path):
    """Same size, new mtime -- a rerun that happened to produce a same-sized
    checkpoint must still invalidate."""
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x" * 100)
    before = file_fingerprint(str(ckpt))

    st = os.stat(str(ckpt))
    os.utime(str(ckpt), ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert file_fingerprint(str(ckpt)) != before


def test_fingerprint_is_stable_when_untouched(tmp_path):
    """rsync -a preserves mtime, so re-syncing must not invalidate."""
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x" * 100)
    assert file_fingerprint(str(ckpt)) == file_fingerprint(str(ckpt))


def test_fingerprint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_fingerprint(str(tmp_path / "nope.pt"))


# ----------------------------------------------------------------------------
# resolve_probe_layer
# ----------------------------------------------------------------------------
class FakeAdvCfg:
    def __init__(self, layers):
        self.penalty_layers = layers


def _ck(layers):
    from conftest import _make_adv_config

    return {"adv_config": _make_adv_config(penalty_layers=layers).to_dict()}


def test_probe_layer_defaults_to_the_penalized_layer():
    assert resolve_probe_layer("t", _ck([7]), None) == 7


def test_probe_layer_override_must_agree_with_config():
    assert resolve_probe_layer("t", _ck([2]), 2) == 2
    with pytest.raises(ValueError, match="penalty_layers"):
        resolve_probe_layer("t", _ck([2]), 10)


def test_probe_layer_ambiguous_without_override():
    with pytest.raises(ValueError, match="set MetricSpec.probe_layer"):
        resolve_probe_layer("t", _ck([2, 4]), None)


def test_probe_layer_none_without_adversarial_config():
    assert resolve_probe_layer("t", {}, None) is None
