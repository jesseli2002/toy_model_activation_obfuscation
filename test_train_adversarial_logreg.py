"""pytest unit tests for the pure-Python helpers in train_adversarial_logreg.py
extracted/cleaned up per plans/train_adversarial_logreg_cleanup_plan.md:
_history_entry, _resolve_hidden_layers, and the Tier-2 --fork-from existence
check (run as a subprocess so it exercises the real pre-heavy-import path).

Also covers the --config/--fork-from machinery from
plans/rare_flags_config_plan.md: load_run_config's missing-required-key
error, and the config.json write/read helpers."""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from config import (
    ForkedFrom,
    LogregAdversarialConfig,
    LogregRunConfig,
    ResidualMLPConfig,
)
from train_adversarial_logreg import (
    TrainRecord,
    _check_config_json,
    _forked_history,
    _history_entry,
    _resolve_hidden_layers,
    _write_run_config,
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


def test_bad_fork_from_tag_exits_before_touching_run_dir(tmp_path):
    """Tier 2's existence check (in the __main__ guard, right after
    parse_args()) must fire before main()'s run-dir setup -- so a
    --fork-from tag with no checkpoint can never let --tag-force delete an
    existing runs/<tag> directory. Runs as a subprocess since the check
    lives in module-level `if __name__ == "__main__":` code."""
    script = Path(__file__).parent / "train_adversarial_logreg.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fork-from",
            "does-not-exist",
            "--tag",
            "unused-test-tag",
            # unused: Tier 2's --fork-from check fires before this is ever
            # read, but parse_args() (Tier 1) requires it to be present.
            "--config",
            str(tmp_path / "unused_config.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr + result.stdout


@pytest.mark.parametrize("mode_args", [["--resume"], ["--fork-from", "some-tag"]])
def test_arch_flag_with_resume_or_fork_from_exits_naming_the_flag(tmp_path, mode_args):
    """Architecture flags are frozen once a checkpoint is being restored --
    passing one explicitly alongside --resume/--fork-from must error (Tier 1,
    in parse_args) rather than being silently ignored."""
    script = Path(__file__).parent / "train_adversarial_logreg.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-x",
            "8",
            "--tag",
            "t",
            "--config",
            str(tmp_path / "unused_config.json"),
            *mode_args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--num-x" in result.stderr + result.stdout


def _file_fields(**overrides) -> dict:
    """A valid --config JSON file's contents (all config-file-only keys)."""
    d = dict(
        penalty_layers=[1, 2],
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
        cfg, hidden_layers = load_run_config(
            _file_fields(), lam=0.7, num_blocks=4, config_path="unused.json"
        )
        assert cfg.lam == 0.7
        assert cfg.penalty_layers == [1, 2]
        assert hidden_layers == [1, 2]
        assert cfg.seed == 1
        assert LogregAdversarialConfig.from_dict(cfg.to_dict()) == cfg

    def test_all_resolves_against_num_blocks(self):
        cfg, hidden_layers = load_run_config(
            _file_fields(penalty_layers="all"),
            lam=0.5,
            num_blocks=4,
            config_path="unused.json",
        )
        assert cfg.penalty_layers == [1, 2, 3]
        assert hidden_layers == [1, 2, 3]

    def test_invalid_penalty_layer_raises_system_exit(self):
        with pytest.raises(SystemExit):
            load_run_config(
                _file_fields(penalty_layers=[0]),
                lam=0.5,
                num_blocks=4,
                config_path="unused.json",
            )

    def test_missing_penalty_layers_raises_system_exit_naming_the_key(self):
        file_fields = _file_fields()
        del file_fields["penalty_layers"]
        with pytest.raises(SystemExit, match="penalty_layers"):
            load_run_config(
                file_fields, lam=0.5, num_blocks=4, config_path="my_config.json"
            )

    def test_missing_key_raises_system_exit_naming_the_key(self):
        file_fields = _file_fields()
        del file_fields["probe_C"]
        with pytest.raises(SystemExit, match="probe_C"):
            load_run_config(
                file_fields, lam=0.5, num_blocks=4, config_path="my_config.json"
            )

    def test_missing_key_error_names_config_path(self):
        file_fields = _file_fields()
        del file_fields["seed"]
        with pytest.raises(SystemExit, match="my_config.json"):
            load_run_config(
                file_fields, lam=0.5, num_blocks=4, config_path="my_config.json"
            )


def _make_adv_config(**overrides) -> LogregAdversarialConfig:
    d = {"lam": 0.5, **_file_fields()}
    d.update(overrides)
    return LogregAdversarialConfig(**d)


def _write_input_config(path: Path, **overrides) -> Path:
    d = _file_fields(**overrides)
    path.write_text(json.dumps(d))
    return path


class TestConfigJsonPersistence:
    """_write_run_config/_check_config_json operate on runs/<tag>/config.json
    (relative paths, via paths.py) -- chdir into tmp_path so these tests don't
    write into the repo's own runs/ directory."""

    def test_write_then_check_matching_config_prints_no_warning(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        model_config = ResidualMLPConfig()
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config("t1", model_config, cfg, input_config_path=str(input_path))
        capsys.readouterr()  # discard _write_run_config's own [config] print
        _check_config_json("t1", model_config, cfg)
        assert "[warn]" not in capsys.readouterr().out

    def test_check_config_json_mismatch_warns_and_leaves_file_untouched(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        model_config = ResidualMLPConfig()
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config("t1", model_config, cfg, input_config_path=str(input_path))
        path = tmp_path / "runs" / "t1" / "config.json"
        on_disk_before = path.read_text()

        capsys.readouterr()  # discard _write_run_config's own [config] print
        different_cfg = _make_adv_config(lam=0.9)
        _check_config_json("t1", model_config, different_cfg)

        assert "[warn]" in capsys.readouterr().out
        assert path.read_text() == on_disk_before  # untouched by the check

    def test_check_config_json_missing_file_warns(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        _check_config_json("t1", ResidualMLPConfig(), _make_adv_config())
        assert "[warn]" in capsys.readouterr().out

    def test_write_run_config_records_forked_from(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t2").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config(
            "t2",
            ResidualMLPConfig(),
            cfg,
            input_config_path=str(input_path),
            forked_from=ForkedFrom(tag="t1", iter=100),
        )
        with open(tmp_path / "runs" / "t2" / "config.json") as f:
            d = json.load(f)
        assert d["forked_from"] == {"tag": "t1", "iter": 100}

    def test_write_run_config_copies_input_file_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runs" / "t3").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config(
            "t3",
            ResidualMLPConfig(),
            _make_adv_config(),
            input_config_path=str(input_path),
        )
        input_copy = tmp_path / "runs" / "t3" / "input_config.json"
        assert input_copy.read_bytes() == input_path.read_bytes()

        resolved = json.loads((tmp_path / "runs" / "t3" / "config.json").read_text())
        assert "lam" not in json.loads(input_copy.read_text())
        assert resolved["adversarial"]["lam"] == 0.5
        assert "model" in resolved
        assert resolved["adversarial"]["penalty_layers"] == [1, 2]


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


class TestAdvConfigCheckpointExtraction:
    """Regression coverage for the reported bug: save_checkpoint used to
    splat adv_config's fields into the checkpoint's top level, alongside
    training-state keys (model/opt/iter/best_loss/probe_w/b/layers) -- so
    LogregAdversarialConfig.from_dict(checkpoint_dict) warned about every
    state key it didn't recognize. Nesting under "adv_config" fixes this."""

    def test_save_checkpoint_shaped_dict_extracts_without_warning(self):
        adv_config = _make_adv_config()
        rck = {
            "model": {},
            "opt": {},
            "iter": 10,
            "best_loss": 0.1,
            "config": {},
            "probe_w": None,
            "probe_b": None,
            "probe_layers": [1, 2],
            "adv_config": adv_config.to_dict(),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            restored = LogregAdversarialConfig.from_dict(rck["adv_config"])
        assert restored == adv_config

    def test_unrecognized_key_inside_adv_config_still_warns(self):
        d = _make_adv_config().to_dict()
        d["_totally_unrecognized_field"] = "surprise"
        with pytest.warns(UserWarning, match="_totally_unrecognized_field"):
            LogregAdversarialConfig.from_dict(d)


def test_resume_missing_adv_config_key_exits_with_error(tmp_path):
    """A checkpoint predating the nested adv_config layout (no "adv_config"
    key) must fail loudly under --resume, not raise a raw KeyError."""
    script = Path(__file__).parent / "train_adversarial_logreg.py"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_file_fields()))
    fresh = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--num-x",
            "2",
            "--d-model",
            "4",
            "--num-blocks",
            "2",
            "--max-iters",
            "1",
            "--tag",
            "t",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert fresh.returncode == 0, fresh.stderr

    import torch

    ckpt_path = tmp_path / "runs" / "t" / "checkpoints" / "last.pt"
    ck = torch.load(ckpt_path, weights_only=False)
    del ck["adv_config"]
    torch.save(ck, ckpt_path)

    result = subprocess.run(
        [sys.executable, str(script), "--resume", "--tag", "t", "--max-iters", "2"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "adv_config" in result.stderr + result.stdout
