"""pytest unit tests for the pure-Python helpers in train_adversarial_logreg.py
extracted/cleaned up per plans/train_adversarial_logreg_cleanup_plan.md:
_history_entry, _resolve_hidden_layers, and the Tier-2 --fork-from existence
check (run as a subprocess so it exercises the real pre-heavy-import path).

Also covers the --config/--fork-from machinery from
plans/rare_flags_config_plan.md: load_run_config's missing-required-key
error, and the config.json write/read helpers."""

import json
import os
import signal
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

from config import ForkedFrom, LogregAdversarialConfig
from conftest import _logreg_config_file_fields, _make_adv_config, _make_model_config
from train_adversarial_logreg import (
    TrainRecord,
    _append_history,
    _check_config_json,
    _defer_keyboard_interrupt,
    _forked_history,
    _history_entry,
    _history_path,
    _read_history,
    _resolve_hidden_layers,
    _restore_rng_state,
    _write_run_config,
    clip_grad_norm_per_block_,
    load_run_config,
    train_steps,
    write_checkpoint,
)


def _make_record(**overrides):
    defaults = dict(
        iter=42,
        loss=1.0,
        l_task=0.5,
        l_probe=0.5,
        l_probe_clean=0.5,
        lam_eff=0.3,
        lr=3e-3,
        affine=("w", "b"),
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
            # unused: Tier 2's --fork-from check fires before these are ever
            # read, but parse_args() (Tier 1) requires them to be present.
            "--config",
            str(tmp_path / "unused_config.json"),
            "--seed",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr + result.stdout


@pytest.mark.parametrize(
    "mode_args", [["--resume"], ["--fork-from", "some-tag", "--seed", "1"]]
)
def test_arch_flag_with_resume_or_fork_from_exits_naming_the_flag(tmp_path, mode_args):
    """Architecture flags are frozen once a checkpoint is being restored --
    passing one explicitly alongside --resume/--fork-from must error (Tier 1,
    in parse_args) rather than being silently ignored. --fork-from's mode_args
    supplies --seed (required for it) so that check doesn't mask this one;
    --resume's leaves it out since --seed is forbidden there."""
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


def test_missing_arch_flag_on_fresh_run_exits_naming_the_flag(tmp_path):
    """num_x/d_model/d_mlp/num_blocks have no default (see ResidualMLPConfig)
    -- a fresh run (no --resume/--fork-from) must name whichever is missing,
    rather than silently falling back to some value."""
    script = Path(__file__).parent / "train_adversarial_logreg.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--d-model",
            "8",
            "--d-mlp",
            "4",
            "--num-blocks",
            "3",
            "--tag",
            "t",
            "--config",
            str(tmp_path / "unused_config.json"),
            "--seed",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--num-x" in result.stderr + result.stdout


class TestLoadRunConfig:
    def test_round_trip(self):
        cfg, hidden_layers = load_run_config(
            _logreg_config_file_fields(lam=0.7),
            num_blocks=4,
            config_path="unused.json",
        )
        assert cfg.lam == 0.7
        assert cfg.penalty_layers == [1, 2]
        assert hidden_layers == [1, 2]
        assert LogregAdversarialConfig.from_dict(cfg.to_dict()) == cfg

    def test_all_resolves_against_num_blocks(self):
        cfg, hidden_layers = load_run_config(
            _logreg_config_file_fields(penalty_layers="all"),
            num_blocks=4,
            config_path="unused.json",
        )
        assert cfg.penalty_layers == [1, 2, 3]
        assert hidden_layers == [1, 2, 3]

    def test_invalid_penalty_layer_raises_system_exit(self):
        with pytest.raises(SystemExit):
            load_run_config(
                _logreg_config_file_fields(penalty_layers=[0]),
                num_blocks=4,
                config_path="unused.json",
            )

    def test_missing_penalty_layers_raises_system_exit_naming_the_key(self):
        file_fields = _logreg_config_file_fields()
        del file_fields["penalty_layers"]
        with pytest.raises(SystemExit, match="penalty_layers"):
            load_run_config(file_fields, num_blocks=4, config_path="my_config.json")

    def test_missing_key_raises_system_exit_naming_the_key(self):
        file_fields = _logreg_config_file_fields()
        del file_fields["probe_C"]
        with pytest.raises(SystemExit, match="probe_C"):
            load_run_config(file_fields, num_blocks=4, config_path="my_config.json")

    def test_missing_key_error_names_config_path(self):
        file_fields = _logreg_config_file_fields()
        del file_fields["lam_warmup_iters"]
        with pytest.raises(SystemExit, match="my_config.json"):
            load_run_config(file_fields, num_blocks=4, config_path="my_config.json")


def _write_input_config(path: Path, **overrides) -> Path:
    d = _logreg_config_file_fields(**overrides)
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
        model_config = _make_model_config()
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config(
            "t1", model_config, cfg, input_config_path=str(input_path), seed=1
        )
        capsys.readouterr()  # discard _write_run_config's own [config] print
        _check_config_json("t1", model_config, cfg)
        assert "[warn]" not in capsys.readouterr().out

    def test_check_config_json_mismatch_warns_and_leaves_file_untouched(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        model_config = _make_model_config()
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t1").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config(
            "t1", model_config, cfg, input_config_path=str(input_path), seed=1
        )
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
        _check_config_json("t1", _make_model_config(), _make_adv_config())
        assert "[warn]" in capsys.readouterr().out

    def test_write_run_config_records_forked_from(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = _make_adv_config()
        (tmp_path / "runs" / "t2").mkdir(parents=True)
        input_path = _write_input_config(tmp_path / "input.json")
        _write_run_config(
            "t2",
            _make_model_config(),
            cfg,
            input_config_path=str(input_path),
            seed=1,
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
            _make_model_config(),
            _make_adv_config(),
            input_config_path=str(input_path),
            seed=1,
        )
        input_copy = tmp_path / "runs" / "t3" / "input_config.json"
        assert input_copy.read_bytes() == input_path.read_bytes()

        resolved = json.loads((tmp_path / "runs" / "t3" / "config.json").read_text())
        assert json.loads(input_copy.read_text())["lam"] == 0.5
        assert resolved["adversarial"]["lam"] == 0.5
        assert "model" in resolved
        assert resolved["adversarial"]["penalty_layers"] == [1, 2]


class TestForkedHistory:
    def test_truncates_at_fork_iter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runs" / "src" / "logs").mkdir(parents=True)
        history = [{"iter": i, "loss": 1.0 / (i + 1)} for i in [0, 10, 20, 30, 40]]
        with open(tmp_path / "runs" / "src" / "logs" / "history.jsonl", "w") as f:
            f.write("\n".join(json.dumps(h) for h in history) + "\n")

        truncated = _forked_history("src", fork_iter=20)
        assert [h["iter"] for h in truncated] == [0, 10, 20]

    def test_missing_history_file_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _forked_history("nonexistent-tag", fork_iter=100) == []


class TestHistoryJsonl:
    def test_append_then_read_round_trips_in_order(self, tmp_path):
        path = tmp_path / "history.jsonl"
        for i in [0, 1, 2]:
            _append_history(str(path), {"iter": i, "loss": 1.0 / (i + 1)})
        assert _read_history(str(path)) == [
            {"iter": 0, "loss": 1.0},
            {"iter": 1, "loss": 0.5},
            {"iter": 2, "loss": 1 / 3},
        ]

    def test_append_does_not_rewrite_existing_lines(self, tmp_path):
        """Regression guard for the quadratic-rewrite bug this format
        replaces: appending must not touch bytes already on disk."""
        path = tmp_path / "history.jsonl"
        _append_history(str(path), {"iter": 0})
        first_write_bytes = path.read_bytes()
        _append_history(str(path), {"iter": 1})
        assert path.read_bytes().startswith(first_write_bytes)

    def test_read_missing_file_returns_empty_list(self, tmp_path):
        assert _read_history(str(tmp_path / "nope.jsonl")) == []

    def test_history_path_uses_jsonl_extension(self):
        assert _history_path("mytag").endswith("history.jsonl")


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
    config_path.write_text(json.dumps(_logreg_config_file_fields()))
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
            "--d-mlp",
            "2",
            "--num-blocks",
            "2",
            "--max-iters",
            "1",
            "--tag",
            "t",
            "--seed",
            "1",
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


def test_resume_missing_rng_state_exits_with_error(tmp_path):
    """A checkpoint predating RNG-state checkpointing (no "rng_state" key)
    must fail loudly under --resume, not raise a raw KeyError."""
    script = Path(__file__).parent / "train_adversarial_logreg.py"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_logreg_config_file_fields()))
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
            "--d-mlp",
            "2",
            "--num-blocks",
            "2",
            "--max-iters",
            "1",
            "--tag",
            "t",
            "--seed",
            "1",
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
    del ck["rng_state"]
    torch.save(ck, ckpt_path)

    result = subprocess.run(
        [sys.executable, str(script), "--resume", "--tag", "t", "--max-iters", "2"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "RNG-state" in result.stderr + result.stdout


def test_restore_rng_state_continues_gen_stream_not_reseeded():
    """_restore_rng_state's `gen` must continue from where the snapshot left
    off (next draw matches the original generator's next draw), not restart
    a fresh stream at whatever seed originally produced that state."""
    import torch

    original_gen = torch.Generator().manual_seed(123)
    original_gen.manual_seed(123)
    torch.randn(5, generator=original_gen)  # advance past the initial state
    snapshot_state = original_gen.get_state()
    expected_next = torch.randn(3, generator=original_gen)

    rck = {
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": None,
        "gen_state": snapshot_state,
    }
    restored_gen = _restore_rng_state(rck, "cpu")
    actual_next = torch.randn(3, generator=restored_gen)

    assert torch.equal(actual_next, expected_next)


class TestNoiseBlobReplay:
    """The property the model-owned noise blob buys over the old gen-state
    snapshot/reset dance (plans/model_noise_blob_plan.md): every forward_loss
    call within one iteration -- including the explode-check and
    explode-redo passes -- sees bit-identical noise, assertable directly
    instead of relying on `gen` replay discipline."""

    def test_explode_check_and_redo_see_identical_noise(self):
        import torch

        from model import ResidualMLP

        model_config = _make_model_config(num_x=2, d_model=4, d_mlp=4, num_blocks=3)
        model = ResidualMLP(model_config)
        opt = torch.optim.AdamW(model.parameters())
        gen = torch.Generator().manual_seed(0)

        adv_config, hidden_layers = load_run_config(
            _logreg_config_file_fields(
                lam=0.0,
                penalty_layers=[1, 2],
                resid_noise_std=0.1,
                explode_factor=1e-6,  # any step counts as an explosion
                batch_size=8,
                lr=1e3,  # huge lr -> guaranteed explode (train_steps sets
                # opt's lr from adv_config every iteration, so this is what
                # actually governs the step size, not opt's own ctor lr above)
            ),
            num_blocks=3,
            config_path="unused.json",
        )

        seen_noise = []
        real_forward = model.forward

        def spy_forward(x_full, *args, noise=None, **kwargs):
            seen_noise.append(noise)
            return real_forward(x_full, *args, noise=noise, **kwargs)

        model.forward = spy_forward

        gen_iter = train_steps(
            model,
            opt,
            gen,
            probe=None,
            adv_config=adv_config,
            max_iters=1,
            hidden_layers=hidden_layers,
            start_iter=0,
            affine=(torch.zeros(1), torch.zeros(1)),
            probe_x=torch.zeros(1, 3),
            probe_label=torch.zeros(1, dtype=torch.bool),
            device="cpu",
        )
        record = next(gen_iter)

        assert record.n_exploded == 1, "test expects the huge-lr step to explode"
        # initial pass, explode-check pass, explode-redo pass.
        assert len(seen_noise) == 3
        for noise in seen_noise[1:]:
            assert torch.equal(seen_noise[0], noise)


class TestExplodeWindow:
    """explode_window_iters (config.py's LogregAdversarialConfig): comparing
    against the smallest loss in a window of recent iterations, not just this
    iteration's own pre-step loss, so a run that creeps up by a sub-threshold
    factor on several consecutive steps still gets caught."""

    def _run(self, monkeypatch, desired_losses, explode_window_iters):
        """Drive train_steps for len(desired_losses)//2-ish iterations with
        model.forward replaced by a stub that returns exactly
        sqrt(desired_losses[call_index]) each call (in call order), against
        an all-zero task target -- so l_task == desired_losses[call_index]
        exactly, independent of real model/gradient dynamics. Returns the
        list of n_exploded seen after each iteration."""
        import torch

        import train_adversarial_logreg as tal
        from model import ResidualMLP

        num_x = 1
        batch_size = 4

        def fake_sample_batch(batch_size, num_x, **kwargs):
            return (
                torch.zeros(batch_size, num_x),
                torch.zeros(batch_size, num_x),
            )

        monkeypatch.setattr(tal, "sample_batch", fake_sample_batch)

        call_idx = [0]

        def fake_forward(x_task, noise=None, **kwargs):
            val = desired_losses[call_idx[0]]
            call_idx[0] += 1
            return torch.full((x_task.shape[0], num_x), val**0.5, requires_grad=True)

        model_config = _make_model_config(num_x=num_x, d_model=4, d_mlp=4, num_blocks=2)
        model = ResidualMLP(model_config)
        model.forward = fake_forward
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(0)

        adv_config, hidden_layers = load_run_config(
            _logreg_config_file_fields(
                lam=0.0,
                penalty_layers=[1],
                batch_size=batch_size,
                explode_factor=2.0,
                explode_window_iters=explode_window_iters,
            ),
            num_blocks=2,
            config_path="unused.json",
        )

        gen_iter = train_steps(
            model,
            opt,
            gen,
            probe=None,
            adv_config=adv_config,
            max_iters=2,
            hidden_layers=hidden_layers,
            start_iter=0,
            affine=(torch.zeros(1), torch.zeros(1)),
            probe_x=torch.zeros(1, 3),
            probe_label=torch.zeros(1, dtype=torch.bool),
            device="cpu",
        )
        return [record.n_exploded for record in gen_iter]

    def test_window_1_never_flags_gradual_creep(self, monkeypatch):
        # Each step's own before/after ratio is 1.9x, under explode_factor=2.0,
        # so a same-iteration-only check (explode_window_iters=1) never fires
        # even though iter 1's loss (3.6) is 3.6x iter 0's starting loss (1.0).
        n_exploded = self._run(
            monkeypatch, desired_losses=[1.0, 1.9, 1.9, 3.6], explode_window_iters=1
        )
        assert n_exploded == [0, 0]

    def test_wider_window_flags_the_same_creep(self, monkeypatch):
        # Same loss trajectory as above, but explode_window_iters=3 compares
        # iter 1's after-step loss (3.6) against the smallest loss over the
        # last 3 iterations (1.0, from iter 0), catching what the 1-step
        # check above missed.
        n_exploded = self._run(
            monkeypatch,
            desired_losses=[1.0, 1.9, 1.9, 3.6, 1.95],
            explode_window_iters=3,
        )
        assert n_exploded == [0, 1]


class TestClipGradNormPerBlock:
    """`clip_grad_norm_per_block_` is a fused rewrite of a per-block
    `clip_grad_norm_` loop, so it is pinned against that loop directly."""

    @staticmethod
    def _blocks(seed=0, widths=((8, 4), (5, 3), (2, 7))):
        g = torch.Generator().manual_seed(seed)
        blocks = []
        for w in widths:
            block = torch.nn.Module()
            block.a = torch.nn.Parameter(torch.randn(*w, generator=g))
            block.b = torch.nn.Parameter(torch.randn(w[0], generator=g))
            blocks.append(block)
        return torch.nn.ModuleList(blocks)

    @staticmethod
    def _set_grads(blocks, seed, scale):
        g = torch.Generator().manual_seed(seed)
        for i, block in enumerate(blocks):
            for p in block.parameters():
                # One block driven far harder, so clipping fires on some
                # blocks and not others -- per-block means the untouched
                # blocks must come through unscaled.
                p.grad = (
                    torch.randn(p.shape, generator=g) * scale * (50.0 if i else 1.0)
                )

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
    def test_matches_per_block_clip_grad_norm_loop(self, scale):
        max_norm = 1.0
        ours = self._blocks()
        theirs = self._blocks()
        self._set_grads(ours, seed=7, scale=scale)
        self._set_grads(theirs, seed=7, scale=scale)

        clip_grad_norm_per_block_(ours, max_norm)
        for block in theirs:
            torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm)

        for a, b in zip(ours.parameters(), theirs.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)

    def test_clips_only_the_blocks_that_exceed_max_norm(self):
        """Guards the property the fused version could most plausibly break:
        one shared scale leaking across block boundaries."""
        blocks = self._blocks()
        # Scaled so block 0 lands under max_norm and the rest well over it.
        self._set_grads(blocks, seed=7, scale=0.1)
        before = [p.grad.clone() for p in blocks.parameters()]
        clip_grad_norm_per_block_(blocks, 1.0)

        norms_before = [
            torch.cat([g.flatten() for g in before[i * 2 : i * 2 + 2]]).norm()
            for i in range(len(blocks))
        ]
        assert norms_before[0] < 1.0 and norms_before[1] > 1.0
        for i, block in enumerate(blocks):
            after = torch.cat([p.grad.flatten() for p in block.parameters()]).norm()
            if norms_before[i] <= 1.0:
                assert after == pytest.approx(norms_before[i].item(), rel=1e-6)
            else:
                assert after == pytest.approx(1.0, rel=1e-5)

    def test_falls_back_when_blocks_have_unequal_grad_counts(self):
        blocks = self._blocks()
        self._set_grads(blocks, seed=7, scale=100.0)
        blocks[1].b.grad = None  # ragged: cannot reshape into a rectangle

        expected = self._blocks()
        self._set_grads(expected, seed=7, scale=100.0)
        expected[1].b.grad = None
        for block in expected:
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)

        clip_grad_norm_per_block_(blocks, 1.0)
        for a, b in zip(blocks.parameters(), expected.parameters()):
            if b.grad is None:
                assert a.grad is None
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)

    def test_no_grads_anywhere_is_a_noop(self):
        clip_grad_norm_per_block_(self._blocks(), 1.0)
        clip_grad_norm_per_block_(torch.nn.ModuleList([]), 1.0)


class TestWriteCheckpoint:
    def _dir(self, tmp_path, saved=None):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        def save_fn(path):
            Path(path).write_text(path)
            if saved is not None:
                saved.append(path)

        return str(ckpt_dir), ckpt_dir, save_fn

    def _numbered(self, ckpt_dir):
        return sorted(int(p.stem[len("iter_") :]) for p in ckpt_dir.glob("iter_*.pt"))

    def test_last_pt_is_a_relative_symlink_to_the_newest_checkpoint(self, tmp_path):
        d, ckpt_dir, save_fn = self._dir(tmp_path)
        write_checkpoint(d, 10, save_fn)
        write_checkpoint(d, 20, save_fn)
        last = ckpt_dir / "last.pt"
        assert os.readlink(last) == "iter_20.pt"  # relative: run dir stays movable
        assert last.resolve() == (ckpt_dir / "iter_20.pt").resolve()

    def test_replaces_a_plain_last_pt_written_by_an_older_run(self, tmp_path):
        d, ckpt_dir, save_fn = self._dir(tmp_path)
        (ckpt_dir / "last.pt").write_text("stale regular file")
        write_checkpoint(d, 10, save_fn)
        assert (ckpt_dir / "last.pt").is_symlink()

    def test_every_checkpoint_is_kept(self, tmp_path):
        d, ckpt_dir, save_fn = self._dir(tmp_path)
        for it in (10, 20, 30, 40):
            write_checkpoint(d, it, save_fn)
        assert self._numbered(ckpt_dir) == [10, 20, 30, 40]
        assert (ckpt_dir / "last.pt").resolve() == (ckpt_dir / "iter_40.pt").resolve()

    def test_writes_through_a_temporary_file(self, tmp_path):
        saved = []
        d, ckpt_dir, save_fn = self._dir(tmp_path, saved=saved)
        write_checkpoint(d, 10, save_fn)
        # the save itself never touches the final path, so a kill mid-write
        # can't truncate a checkpoint last.pt might already point at
        assert saved == [str(ckpt_dir / "iter_10.pt.tmp")]
        assert not list(ckpt_dir.glob("*.tmp"))

    def test_never_removes_a_file_that_is_not_a_numbered_checkpoint(self, tmp_path):
        d, ckpt_dir, save_fn = self._dir(tmp_path)
        (ckpt_dir / "best.pt").write_text("best")
        os.symlink("best.pt", ckpt_dir / "last.pt")
        write_checkpoint(d, 10, save_fn)
        assert (ckpt_dir / "best.pt").exists()


class TestDeferKeyboardInterrupt:
    def test_sigint_is_recorded_and_re_raised_after_the_block(self):
        ran_to_completion = False
        with pytest.raises(KeyboardInterrupt):
            with _defer_keyboard_interrupt() as sigint:
                assert not sigint.interrupted
                os.kill(os.getpid(), signal.SIGINT)
                assert sigint.interrupted  # deferred, not raised in place
                os.kill(os.getpid(), signal.SIGINT)  # spamming changes nothing
                ran_to_completion = True
        assert ran_to_completion

    def test_restores_the_previous_handler(self):
        before = signal.getsignal(signal.SIGINT)
        with _defer_keyboard_interrupt():
            pass
        assert signal.getsignal(signal.SIGINT) is before
