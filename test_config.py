"""pytest unit tests for the shared _CheckpointConfigMixin to_dict/from_dict
backfill idiom in config.py, parametrized across all three config
dataclasses that use it."""

import dataclasses

import pytest

from config import (
    AdversarialConfig,
    ForkedFrom,
    LogregAdversarialConfig,
    LogregRunConfig,
    ResidualMLPConfig,
)
from conftest import _make_logreg_config, _make_model_config

# (config class, a field whose _LEGACY_DEFAULTS value diverges from the
# dataclass field default -- used to distinguish "backfilled from legacy"
# from "backfilled from the field default", factory building a valid instance).
CONFIGS_WITH_DIVERGENT_FIELD = [
    (ResidualMLPConfig, "num_blocks", _make_model_config),
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

    def test_lam_has_a_default(self):
        # the one CLI-common field must NOT require a value.
        kwargs = dataclasses.asdict(_make_logreg_config())
        del kwargs["lam"]
        LogregAdversarialConfig(**kwargs)  # must not raise

    def test_missing_penalty_layers_raises_type_error(self):
        kwargs = dataclasses.asdict(_make_logreg_config())
        del kwargs["penalty_layers"]
        with pytest.raises(TypeError):
            LogregAdversarialConfig(**kwargs)


class TestLogregRunConfig:
    def test_round_trip_without_forked_from(self):
        run_config = LogregRunConfig(
            model=_make_model_config(), adversarial=_make_logreg_config()
        )
        assert LogregRunConfig.from_dict(run_config.to_dict()) == run_config

    def test_forked_from_absent_from_dict_when_none(self):
        run_config = LogregRunConfig(
            model=_make_model_config(), adversarial=_make_logreg_config()
        )
        assert "forked_from" not in run_config.to_dict()

    def test_forked_from_round_trips_when_present(self):
        run_config = LogregRunConfig(
            model=_make_model_config(),
            adversarial=_make_logreg_config(),
            forked_from=ForkedFrom(tag="source", iter=100),
        )
        d = run_config.to_dict()
        assert d["forked_from"] == {"tag": "source", "iter": 100}
        assert LogregRunConfig.from_dict(d) == run_config

    def test_legacy_backfill_applies_per_block(self):
        """A field missing from the nested `model`/`adversarial` dicts backfills
        through each block's own _LEGACY_DEFAULTS, same as calling that
        dataclass's from_dict directly."""
        run_config = LogregRunConfig(
            model=_make_model_config(), adversarial=_make_logreg_config()
        )
        d = run_config.to_dict()
        del d["model"]["num_blocks"]
        del d["adversarial"]["probe_retrain_interval"]
        restored = LogregRunConfig.from_dict(d)
        assert (
            restored.model.num_blocks
            == ResidualMLPConfig._LEGACY_DEFAULTS["num_blocks"]
        )
        assert (
            restored.adversarial.probe_retrain_interval
            == LogregAdversarialConfig._LEGACY_DEFAULTS["probe_retrain_interval"]
        )
