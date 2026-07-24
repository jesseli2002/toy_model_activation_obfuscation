"""pytest unit tests for the pure-Python helpers in train_adversarial_logreg.py
extracted/cleaned up per plans/train_adversarial_logreg_cleanup_plan.md:
_history_entry, _resolve_hidden_layers, and the Tier-2 --warmstart existence
check (run as a subprocess so it exercises the real pre-heavy-import path)."""

import subprocess
import sys
from pathlib import Path

import pytest

from train_adversarial_logreg import TrainRecord, _history_entry, _resolve_hidden_layers


def _make_record(**overrides):
    defaults = dict(
        it=42,
        loss=1.0,
        l_task=0.5,
        l_probe=0.5,
        lam_eff=0.3,
        affine=("w", "b"),
        probe_dt=0.1,
        model_dt=0.2,
    )
    defaults.update(overrides)
    return TrainRecord(**defaults)


class TestHistoryEntry:
    def test_no_affine_key(self):
        d = _history_entry(_make_record())
        assert "affine" not in d

    def test_iter_key_preserved_for_adversarial_report(self):
        # adversarial_report.py reads history[...]["iter"], not "it" (the
        # TrainRecord field name) -- must not silently rename this key.
        d = _history_entry(_make_record(it=7))
        assert d["iter"] == 7
        assert "it" not in d

    def test_extra_adds_new_keys(self):
        d = _history_entry(_make_record(), max_err=1e-3)
        assert d["max_err"] == 1e-3

    def test_extra_overrides_colliding_keys(self):
        d = _history_entry(_make_record(loss=1.0), loss=0.5)
        assert d["loss"] == 0.5


class TestResolveHiddenLayers:
    def test_all_resolves_to_1_through_num_blocks_minus_1(self):
        assert _resolve_hidden_layers("all", 4) == [1, 2, 3]

    def test_layer_zero_raises_system_exit(self):
        with pytest.raises(SystemExit):
            _resolve_hidden_layers([0, 1], 4)

    def test_final_layer_warns_and_proceeds(self, capsys):
        layers = _resolve_hidden_layers([4], 4)
        assert layers == [4]
        assert "warn" in capsys.readouterr().out

    def test_out_of_range_raises_system_exit(self):
        with pytest.raises(SystemExit):
            _resolve_hidden_layers([5], 4)


def test_bad_warmstart_path_exits_before_touching_run_dir(tmp_path):
    """Tier 2's existence check (in the __main__ guard, right after
    parse_args()) must fire before main()'s run-dir setup -- so a bad
    --warmstart path can never let --tag-force delete an existing
    runs/<tag> directory. Runs as a subprocess since the check lives in
    module-level `if __name__ == "__main__":` code."""
    script = Path(__file__).parent / "train_adversarial_logreg.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--warmstart",
            str(tmp_path / "does_not_exist.pt"),
            "--tag",
            "unused-test-tag",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr + result.stdout
