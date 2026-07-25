"""pytest unit tests for the shared _CheckpointConfigMixin to_dict/from_dict
backfill idiom in config.py, parametrized across all three config
dataclasses that use it."""

import dataclasses

import pytest

from config import AdversarialConfig, LogregAdversarialConfig, ResidualMLPConfig


def _make_logreg_config() -> LogregAdversarialConfig:
    """LogregAdversarialConfig's config-file-only fields have no Python
    default (see plans/rare_flags_config_plan.md) -- unlike the other two
    config classes below, `LogregAdversarialConfig()` alone isn't a valid
    construction, so the generic tests need a fully-specified instance."""
    return LogregAdversarialConfig(
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


# (config class, a field whose _LEGACY_DEFAULTS value diverges from the
# dataclass field default -- used to distinguish "backfilled from legacy"
# from "backfilled from the field default", factory building a valid instance).
CONFIGS_WITH_DIVERGENT_FIELD = [
    (ResidualMLPConfig, "num_blocks", ResidualMLPConfig),
    (AdversarialConfig, "probe_loss", AdversarialConfig),
    (LogregAdversarialConfig, "probe_retrain_interval", _make_logreg_config),
]


@pytest.mark.parametrize("cls, divergent_field, make", CONFIGS_WITH_DIVERGENT_FIELD)
class TestCheckpointConfigMixin:
    def test_round_trip(self, cls, divergent_field, make):
        assert cls.from_dict(make().to_dict()) == make()

    def test_legacy_backfill_uses_legacy_default_not_field_default(
        self, cls, divergent_field, make
    ):
        legacy_value = cls._LEGACY_DEFAULTS[divergent_field]
        field_default = getattr(make(), divergent_field)
        assert legacy_value != field_default, (
            f"{cls.__name__}.{divergent_field} was chosen as a divergent "
            "field but its legacy/field defaults coincide -- pick a field "
            "that actually diverges"
        )

        d = make().to_dict()
        del d[divergent_field]
        backfilled = cls.from_dict(d)
        assert getattr(backfilled, divergent_field) == legacy_value

    def test_unknown_key_warns_and_is_dropped(self, cls, divergent_field, make):
        d = make().to_dict()
        d["_totally_unrecognized_field"] = "surprise"

        with pytest.warns(UserWarning, match="_totally_unrecognized_field"):
            reconstructed = cls.from_dict(d)

        assert not hasattr(reconstructed, "_totally_unrecognized_field")
        assert reconstructed == make()


class TestLogregAdversarialConfigRequiredFields:
    """LogregAdversarialConfig's config-file-only fields have no default, so
    omitting one is a TypeError -- the mechanism
    train_adversarial_logreg.load_run_config relies on to fail loudly on a
    --config file missing a required key (see plans/rare_flags_config_plan.md)."""

    def test_missing_required_field_raises_type_error(self):
        kwargs = dataclasses.asdict(_make_logreg_config())
        del kwargs["seed"]
        with pytest.raises(TypeError):
            LogregAdversarialConfig(**kwargs)

    def test_lam_and_penalty_layers_have_defaults(self):
        # the two CLI-common fields must NOT require a value.
        kwargs = dataclasses.asdict(_make_logreg_config())
        del kwargs["lam"]
        del kwargs["penalty_layers"]
        LogregAdversarialConfig(**kwargs)  # must not raise
