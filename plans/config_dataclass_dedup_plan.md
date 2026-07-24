# Config dataclass deduplication

## Context

`config.py` has three dataclasses (`ResidualMLPConfig`, `AdversarialConfig`,
`LogregAdversarialConfig`) that are stored verbatim (as dicts) in checkpoints
and must survive old checkpoints gaining new fields over time. Each
independently implements the *same* idiom:

- a per-field dataclass default (what a fresh `Config(...)` gets if the field
  is omitted),
- a separate `_LEGACY_DEFAULTS` `ClassVar[dict]` (what an *old* checkpoint --
  saved before the field existed -- backfills to), which deliberately diverges
  from the dataclass default in several cases (e.g. `ResidualMLPConfig`
  forward-defaults `num_blocks=8` but `_LEGACY_DEFAULTS["num_blocks"] = 4`,
  since old checkpoints were trained with 4 blocks),
- an identical `to_dict`/`from_dict` pair (~15 lines), including the same
  unrecognized-key warning.

This is pure mechanism duplication -- the fields differ across the three
classes, the backfill machinery doesn't. Each class's docstring also
re-explains the mechanism itself in a full paragraph, so the explanation is
tripled too. Found during the `train_adversarial_logreg.py` tech-debt review;
this item is independent of the other two plans from that review and can land
in any order relative to them.

## Design

Extract a small non-dataclass mixin that owns `to_dict`/`from_dict`. It reads
`_LEGACY_DEFAULTS` and `dataclasses.fields(cls)` off whichever subclass calls
it (via `cls`), so it needs no fields of its own -- no dataclass
field-ordering concerns from mixing it into `@dataclass` subclasses.

```python
class _CheckpointConfigMixin:
    """Shared to_dict/from_dict for config dataclasses stored verbatim in
    checkpoints. Subclasses must be @dataclass and define a ClassVar
    _LEGACY_DEFAULTS dict covering every optional field, used to backfill
    checkpoints saved before that field existed. See e.g.
    ResidualMLPConfig's docstring for why this differs from the plain
    dataclass field default."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            warnings.warn(
                f"{cls.__name__}.from_dict: dropping unrecognized key(s) "
                f"{sorted(unknown)} -- checkpoint saved by a newer version?"
            )
        present = {k: v for k, v in d.items() if k in known}
        return cls(**(cls._LEGACY_DEFAULTS | present))
```

Docs: the mixin's docstring owns the general "two notions of default, and why
they diverge" explanation once. Each subclass's docstring keeps only what's
specific to it -- what the class is for, and which of its own fields actually
diverge between forward-default and legacy-default (e.g. `num_blocks`
8-vs-4, `probe_loss` "lda"-vs-"squared") -- with a pointer back to the mixin
for the general mechanism instead of re-deriving it.

## Steps

1. Add `_CheckpointConfigMixin` to `config.py`, above the three dataclasses.
2. `class ResidualMLPConfig(_CheckpointConfigMixin): ...` -- remove its own
   `to_dict`/`from_dict`, trim its docstring to the class-specific
   divergence example, pointer to the mixin for the mechanism.
3. Same for `AdversarialConfig` and `LogregAdversarialConfig`.
4. Grep for any other direct use of the removed per-class `to_dict`/`from_dict`
   definitions to confirm nothing depended on class-specific override
   behavior (expected: none -- all three were identical).

## Verification

- New `test_config.py`, one parametrized test class covering all three
  dataclasses:
  - round-trip: `Config.from_dict(Config().to_dict()) == Config()`.
  - legacy backfill: a dict missing a field that diverges between forward
    and legacy default (e.g. `ResidualMLPConfig` without `"num_blocks"`)
    backfills to `_LEGACY_DEFAULTS[field]`, not the dataclass field default.
  - unknown-key handling: a dict with an extra unrecognized key trips
    `warnings.warn` (`pytest.warns`) and the key is absent from the
    reconstructed object.
- Load an existing real checkpoint (e.g. the canonical `runs/nx32` checkpoint)
  through `ResidualMLP.load` before and after the refactor and diff the
  resulting `model.config` -- must be identical.
- `black --check config.py`.
- `python -m pytest test_torch_logreg.py` (unaffected, but cheap regression
  check that nothing else in the module broke on import).

## Risks / caveats

Low risk -- this is a pure refactor with identical external behavior for all
three classes (they were byte-identical in mechanism already). The only
failure mode worth watching for is `cls._LEGACY_DEFAULTS` resolving to the
wrong class's dict if the mixin is ever called unbound from a subclass
instance rather than as a classmethod; `dataclasses.fields(cls)` and
`cls._LEGACY_DEFAULTS` both rely on Python's normal classmethod `cls`
polymorphism, which is already how the pre-refactor code worked, so this is
not a new risk, just worth a comment.
