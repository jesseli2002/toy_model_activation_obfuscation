"""pytest unit tests for the shared _CheckpointConfigMixin to_dict/from_dict
backfill idiom in config.py, parametrized across all three config
dataclasses that use it."""

import pytest

from config import AdversarialConfig, LogregAdversarialConfig, ResidualMLPConfig

# (config class, a field whose _LEGACY_DEFAULTS value diverges from the
# dataclass field default -- used to distinguish "backfilled from legacy"
# from "backfilled from the field default").
CONFIGS_WITH_DIVERGENT_FIELD = [
    (ResidualMLPConfig, "num_blocks"),
    (AdversarialConfig, "probe_loss"),
    (LogregAdversarialConfig, "probe_retrain_interval"),
]


@pytest.mark.parametrize("cls, divergent_field", CONFIGS_WITH_DIVERGENT_FIELD)
class TestCheckpointConfigMixin:
    def test_round_trip(self, cls, divergent_field):
        assert cls.from_dict(cls().to_dict()) == cls()

    def test_legacy_backfill_uses_legacy_default_not_field_default(
        self, cls, divergent_field
    ):
        legacy_value = cls._LEGACY_DEFAULTS[divergent_field]
        field_default = getattr(cls(), divergent_field)
        assert legacy_value != field_default, (
            f"{cls.__name__}.{divergent_field} was chosen as a divergent "
            "field but its legacy/field defaults coincide -- pick a field "
            "that actually diverges"
        )

        d = cls().to_dict()
        del d[divergent_field]
        backfilled = cls.from_dict(d)
        assert getattr(backfilled, divergent_field) == legacy_value

    def test_unknown_key_warns_and_is_dropped(self, cls, divergent_field):
        d = cls().to_dict()
        d["_totally_unrecognized_field"] = "surprise"

        with pytest.warns(UserWarning, match="_totally_unrecognized_field"):
            reconstructed = cls.from_dict(d)

        assert not hasattr(reconstructed, "_totally_unrecognized_field")
        assert reconstructed == cls()
