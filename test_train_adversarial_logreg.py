"""pytest unit tests for the pure-Python helpers in train_adversarial_logreg.py
extracted/cleaned up per plans/train_adversarial_logreg_cleanup_plan.md:
_history_entry, _resolve_hidden_layers, and the Tier-2 --warmstart existence
check (run as a subprocess so it exercises the real pre-heavy-import path).

Also covers the --config/--fork-from machinery from
plans/rare_flags_config_plan.md: load_run_config's missing-required-key
error, and the config.json write/read helpers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from config import LogregAdversarialConfig
from train_adversarial_logreg import (
    TrainRecord,
    _check_config_json,
    _forked_history,
    _history_entry,
    _resolve_hidden_layers,
    _write_config_json,
    load_run_config,
)


def _make_record(**overrides):
    defaults = dict(
        iter=42,
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

    def test_iter_key_matches_adversarial_report_expectations(self):
        # adversarial_report.py reads history[...]["iter"] -- TrainRecord's
        # field is named `iter` (not `it`) precisely so asdict() lines up
        # with this on-disk schema without any renaming.
        d = _history_entry(_make_record(iter=7))
        assert d["iter"] == 7

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
            # unused: Tier 2's --warmstart check fires before this is ever
            # read, but parse_args() (Tier 1) requires it to be present.
            "--config",
            str(tmp_path / "unused_config.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr + result.stdout


def _file_fields(**overrides) -> dict:
    """A valid --config JSON file's contents (all 14 config-file-only keys)."""
    d = dict(
        lam_warmup_iters=0,
        seed=1,
        probe_C=1.0,
        probe_init_iters=1000,
        class_threshold=1.5,
        probe_loss_kind="meandiff-relu",
        probe_subsample=8,
        probe_retrain_interval=16,
        resid_noise_std=0.1,
        grad_clip=1.0,
        x_p_outer=None,
        x_threshold=1.0,
        batch_size=4096,
        lr=3e-3,
        adam_eps=1e-8,
        adam_beta2=0.999,
        explode_factor=0.0,
        explode_clip_divisor=5.0,
    )
    d.update(overrides)
    return d


class TestLoadRunConfig:
    def test_round_trip(self):
        cfg = load_run_config(
            _file_fields(), lam=0.7, penalty_layers=[1, 2], config_path="unused.json"
        )
        assert cfg.lam == 0.7
        assert cfg.penalty_layers == [1, 2]
        assert cfg.seed == 1
        assert LogregAdversarialConfig.from_dict(cfg.to_dict()) == cfg

    def test_missing_key_raises_system_exit_naming_the_key(self):
        file_fields = _file_fields()
        del file_fields["probe_C"]
        with pytest.raises(SystemExit, match="probe_C"):
            load_run_config(
                file_fields, lam=0.5, penalty_layers=[1], config_path="my_config.json"
            )

    def test_missing_key_error_names_config_path(self):
        file_fields = _file_fields()
        del file_fields["seed"]
        with pytest.raises(SystemExit, match="my_config.json"):
            load_run_config(
                file_fields, lam=0.5, penalty_layers=[1], config_path="my_config.json"
            )


def _make_adv_config(**overrides) -> LogregAdversarialConfig:
    d = {"lam": 0.5, "penalty_layers": [1, 2], **_file_fields()}
    d.update(overrides)
    return LogregAdversarialConfig(**d)


class TestConfigJsonPersistence:
    """_write_config_json/_check_config_json operate on runs/<tag>/config.json
    (relative paths, via paths.py) -- chdir into tmp_path so these tests don't
    write into the repo's own runs/ directory."""

    def test_write_then_check_matching_config_prints_no_warning(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        _write_config_json("t1", cfg)
        capsys.readouterr()  # discard _write_config_json's own [config] print
        _check_config_json("t1", cfg)
        assert "[warn]" not in capsys.readouterr().out

    def test_check_config_json_mismatch_warns_and_leaves_file_untouched(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        _write_config_json("t1", cfg)
        path = tmp_path / "runs" / "t1" / "config.json"
        on_disk_before = path.read_text()

        capsys.readouterr()  # discard _write_config_json's own [config] print
        different_cfg = _make_adv_config(lam=0.9)
        _check_config_json("t1", different_cfg)

        assert "[warn]" in capsys.readouterr().out
        assert path.read_text() == on_disk_before  # untouched by the check

    def test_write_config_json_records_forked_from(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t2").mkdir(parents=True)
        _write_config_json("t2", cfg, forked_from={"tag": "t1", "iter": 100})
        with open(tmp_path / "runs" / "t2" / "config.json") as f:
            d = json.load(f)
        assert d["forked_from"] == {"tag": "t1", "iter": 100}


class TestForkedHistory:
    def test_truncates_at_fork_iter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runs" / "src" / "logs").mkdir(parents=True)
        history = [{"iter": i, "loss": 1.0 / (i + 1)} for i in [0, 10, 20, 30, 40]]
        with open(tmp_path / "runs" / "src" / "logs" / "history.json", "w") as f:
            json.dump(history, f)

        truncated = _forked_history("src", fork_iter=20)
        assert [h["iter"] for h in truncated] == [0, 10, 20]

    def test_missing_history_file_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _forked_history("nonexistent-tag", fork_iter=100) == []
