"""Shared config-fixture builders for test_config.py and
test_train_adversarial_logreg.py -- kept in one place so that adding a new
required field to a config dataclass (e.g. LogregAdversarialConfig's
config-file-only fields, see plans/rare_flags_config_plan.md) only needs
updating here instead of at every test call site."""

from config import LogregAdversarialConfig, ResidualMLPConfig


def _make_model_config(**overrides) -> ResidualMLPConfig:
    """ResidualMLPConfig's num_x/d_model/d_mlp/num_blocks have no Python
    default (every caller must pin the architecture explicitly), so the
    generic tests need a fully-specified instance."""
    d = dict(num_x=4, d_model=8, d_mlp=4, num_blocks=3)
    d.update(overrides)
    return ResidualMLPConfig(**d)


def _logreg_config_file_fields(**overrides) -> dict:
    """A valid --config JSON file's contents: every LogregAdversarialConfig
    field that has no Python default (see plans/rare_flags_config_plan.md),
    so `LogregAdversarialConfig()` alone isn't a valid construction and
    tests need a fully-specified set of fields."""
    d = dict(
        lam=0.5,
        penalty_layers=[1, 2],
        lam0_warmup_iters=0,
        lam_warmup_iters=0,
        probe_C=1.0,
        probe_init_iters=1000,
        class_threshold=1.5,
        probe_loss_kind="meandiff-relu",
        probe_subsample=8,
        probe_retrain_interval=16,
        probe_resample_interval=512,
        probe_loss_trim_frac=0.05,
        resid_noise_std=0.1,
        probe_noise=False,
        grad_clip=1.0,
        x_p_outer=None,
        x_threshold=1.0,
        batch_size=4096,
        lr=3e-3,
        lr_warmup_iters=0,
        lr_min_frac=1.0,
        lr_schedule="cosine",
        adam_eps=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        explode_factor=0.0,
        explode_clip_divisor=5.0,
        explode_window_iters=1,
        optimizer_kind="adamw",
        stableadamw_d=1.0,
    )
    d.update(overrides)
    return d


def _make_logreg_config(**overrides) -> LogregAdversarialConfig:
    return LogregAdversarialConfig(**_logreg_config_file_fields(**overrides))


def _make_adv_config(**overrides) -> LogregAdversarialConfig:
    d = _logreg_config_file_fields()
    d.update(overrides)
    return LogregAdversarialConfig(**d)
