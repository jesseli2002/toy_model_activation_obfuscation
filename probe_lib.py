"""Utilities for probe analysis: capture residual-stream activations from a
checkpoint, build eval sets over them, fit/compare/score linear decision
boundaries, and plot the result. (See checkpoint_lib.py for loading a
checkpoint in the first place -- this module builds on top of that.)

Used as a library by the reporting scripts (adversarial_report.py and the
sweep_*.py family) and by analytic.py (activation capture for hand-built
exact models). Anything a report needs in order to turn a checkpoint into a
probe score belongs here rather than in one of those entry points, so the
reports agree on what "probe AUROC" means.
"""

import os
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

# load_model/load_model_path/resolve_adv_config re-exported from
# checkpoint_lib for existing callers; see that module for their definitions.
from checkpoint_lib import load_model, load_model_path, resolve_adv_config
from config import C_HIGH, C_LOW, PROBE_EVAL_N_TEST, PROBE_EVAL_N_TRAIN
from data import sample_batch, sample_fixed_c
from model import Noise, ResidualMLP


class LinearBoundary(NamedTuple):
    """An affine decision boundary over probed features: score(x) = w . x + b,
    boundary at score = 0. Used to compare probes (e.g. difference-of-means vs
    a fit logreg/LDA probe) that share this representation regardless of how
    each was derived."""

    w: Float[np.ndarray, "d"]
    b: float

    def score(self, X: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n"]:
        return X @ self.w + self.b


class BinaryEvalSet(NamedTuple):
    """A fixed two-class contrast to score probes against: class 0 is c=c_lo,
    class 1 is c=c_hi, `n` inputs each.

    Deliberately a fixed pair rather than the continuous c ~ U[C_LOW, C_HIGH]
    training distribution -- sampled once and reused, so scores stay
    comparable point to point across checkpoints and across runs."""

    x: Float[torch.Tensor, "n2 num_x_plus_1"]
    y: Bool[np.ndarray, "n2"]


def make_binary_eval_set(
    num_x: int,
    n: int,
    c_lo: float = C_LOW,
    c_hi: float = C_HIGH,
    *,
    generator: torch.Generator,
    device: str,
) -> BinaryEvalSet:
    """`n` inputs per class (so 2n rows total, the c_lo half first)."""
    xf_lo, _ = sample_fixed_c(n, num_x, c_lo, generator=generator, device=device)
    xf_hi, _ = sample_fixed_c(n, num_x, c_hi, generator=generator, device=device)
    return BinaryEvalSet(
        x=torch.cat([xf_lo, xf_hi], dim=0),
        y=np.concatenate([np.zeros(n), np.ones(n)]),
    )


def boundary_accuracy(
    b: LinearBoundary, X: Float[np.ndarray, "n d"], y: Bool[np.ndarray, "n"]
) -> float:
    """Fraction of (X, y) classified correctly by `b.score(X) > 0`."""
    return float(((b.score(X) > 0) == y).mean())


def boundary_auroc(
    b: LinearBoundary, X: Float[np.ndarray, "n d"], y: Bool[np.ndarray, "n"]
) -> float:
    """AUROC of `b`'s scores against `y`. Threshold-free, so unlike
    `boundary_accuracy` it ignores the bias term."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, b.score(X)))


@torch.no_grad()
def capture_layers_dict(
    model: ResidualMLP,
    x_full: torch.Tensor,
    layers: list[int],
    noise: float | torch.Tensor = 0.0,
    generator: torch.Generator | None = None,
) -> dict[int, torch.Tensor]:
    """Returns each requested layer's residual-stream activations, keyed by
    layer index, from a single shared forward pass. `noise`/`generator` are
    passed straight through to `ResidualMLP.forward` (as its `noise`/
    `generator` params) -- a float draws a fresh std everywhere, a tensor
    (from `ResidualMLP.generate_noise`) injects that exact noise, letting a
    caller vary the std per injection point."""
    _, caches = model.forward(
        x_full, return_cache=True, noise=noise, generator=generator
    )
    return {i: caches[i] for i in layers}


@torch.no_grad()
def capture_layers(
    model: ResidualMLP,
    x_full: torch.Tensor,
    layers: list[int],
    noise: float | torch.Tensor = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Like capture_layers_dict, but concatenates the requested layers into a
    single flat feature tensor."""
    d = capture_layers_dict(model, x_full, layers, noise=noise, generator=generator)
    return torch.cat([d[i] for i in layers], dim=-1)


def stored_probe(ck, device: str) -> tuple[LinearBoundary, list[int]] | None:
    """The probe fit during training and saved alongside the model weights, as
    (boundary, probed layers). None for a checkpoint carrying no probe.

    Contrast the refit path (`binary_probe_metrics_all_layers`), which fits a
    fresh probe against the checkpoint instead: this one answers "how visible
    was c to the adversary the model was actually trained against"."""
    if "probe_w" not in ck:
        return None
    w = ck["probe_w"].to(device).cpu().numpy()
    b = float(ck["probe_b"])
    return LinearBoundary(w, b), list(ck["probe_layers"])


@torch.no_grad()
def stored_probe_auroc(
    model: ResidualMLP,
    ck: dict,
    eval_set: BinaryEvalSet,
    *,
    eval_noise_mult: float = 1.0,
    generator: torch.Generator | None = None,
) -> float | None:
    """AUROC of the checkpoint's own training-time probe over `eval_set`, or
    None when the checkpoint carries no probe.

    The eval forward pass gets `eval_noise_mult` * the run's own training-time
    resid_noise_std of residual-stream noise, so a report can ask how the
    probe holds up under more or less noise than the model trained with."""
    stored = stored_probe(ck, str(next(model.parameters()).device))
    if stored is None:
        return None
    boundary, layers = stored
    adv_cfg = resolve_adv_config(ck)
    # A stored probe means this came from train_adversarial_logreg's
    # save_checkpoint, which always writes adv_config alongside it -- fail
    # loudly rather than quietly scoring an unexpected checkpoint at 0 noise.
    assert adv_cfg is not None, "checkpoint has a stored probe but no adv_config"
    feats = capture_layers(
        model,
        eval_set.x,
        layers,
        noise=adv_cfg.resid_noise_std * eval_noise_mult,
        generator=generator,
    )
    return boundary_auroc(boundary, feats.cpu().numpy(), eval_set.y)


class StoredProbeScorer:
    """Scores many runs' stored probes against one shared eval set.

    The eval set is built lazily from the first checkpoint loaded (the runs in
    a sweep share an architecture, so any of them fixes num_x) and reused for
    every subsequent run, which is what makes the resulting AUROCs comparable
    across a sweep. Owns the one RNG the whole thing needs -- both the eval
    sampling and the per-run eval noise draw from it."""

    def __init__(
        self,
        ckpt: str,
        device: str,
        *,
        eval_n: int = 4096,
        eval_noise_mult: float = 1.0,
        seed: int = 0,
    ):
        self.ckpt = ckpt
        self.device = device
        self.eval_n = eval_n
        self.eval_noise_mult = eval_noise_mult
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.eval_set: BinaryEvalSet | None = None

    def auroc(self, tag: str) -> float | None:
        """AUROC of `tag`'s stored probe, or None if it has no stored probe."""
        model, ck = load_model(tag, self.ckpt, self.device)
        if self.eval_set is None:
            self.eval_set = make_binary_eval_set(
                model.config.num_x, self.eval_n, generator=self.g, device=self.device
            )
        return stored_probe_auroc(
            model,
            ck,
            self.eval_set,
            eval_noise_mult=self.eval_noise_mult,
            generator=self.g,
        )


# ----------------------------------------------------------------------------
# Refit-probe path: fit a fresh probe against a checkpoint
# ----------------------------------------------------------------------------
# Regularization constant for LogisticRegression (see scikit-learn
# LogisticRegression interface). Larger = less regularization. Deliberately
# weak, to give the probe every chance and keep the hiding result conservative.
PROBE_C = 1000

# Analysis refits solve in float64 regardless of the model's own dtype. `C`
# multiplies the SUMMED loss, so the Newton Hessian scales as C * n while the
# regularizer contributes O(1); by n_train ~ 30k that spread exceeds float32's
# resolution and the solve silently stops converging -- ||w|| runs away and
# AUROC collapses toward chance, i.e. it understates probe strength exactly
# where a bigger fit set was supposed to help. float64 removes it for ~0.4s a
# fit. Training is unaffected: at probe_C=1 on a subsampled batch it sits four
# orders of magnitude away from this regime. Measurements in PR #239.
PROBE_EVAL_FIT_DTYPE = torch.float64


def probe_eval_solver_key() -> str:
    """Identifies the settings a refit probe was solved under, for callers that
    cache results keyed by everything a value depends on -- so changing the fit
    precision or step budget invalidates entries rather than silently reusing
    them."""
    from probe_backend import PROBE_EVAL_NEWTON_COLD_STEPS

    return f"{PROBE_EVAL_FIT_DTYPE}/cold{PROBE_EVAL_NEWTON_COLD_STEPS}"


def raw_signed_distance(
    w_probe: Float[np.ndarray, "d"], b_probe: float, X: Float[np.ndarray, "n d"]
) -> Float[np.ndarray, "n"]:
    """Signed distance to the probe's decision boundary in raw
    (unstandardized) data units, boundary at 0 -- same fold as `plot_probe`'s
    raw-space panel."""
    w_hat = w_probe / np.linalg.norm(w_probe)
    threshold = -b_probe / np.linalg.norm(w_probe)
    return X @ w_hat - threshold


@torch.no_grad()
def binary_dataset_all_layers(
    model, num_x, n, c_lo, c_hi, layers, generator, device, noise=0.0
):
    """One forward pass over c_lo and one over c_hi, shared across all
    `layers` -- returns {layer: (r_lo, r_hi)}. `noise` (see
    `capture_layers_dict`) injects residual-stream noise into both forward
    passes, same as the model's own training-time environment. A tensor is
    reused verbatim across the c_lo and c_hi passes (rather than redrawn) --
    harmless since each pass's noise is i.i.d. per example either way, and
    the two passes aren't paired by index."""
    xf_lo, _ = sample_fixed_c(n, num_x, c_lo, generator=generator, device=device)
    xf_hi, _ = sample_fixed_c(n, num_x, c_hi, generator=generator, device=device)
    r_lo = capture_layers_dict(model, xf_lo, layers, noise=noise, generator=generator)
    r_hi = capture_layers_dict(model, xf_hi, layers, noise=noise, generator=generator)
    return {layer: (r_lo[layer], r_hi[layer]) for layer in layers}


@torch.no_grad()
def linear_y_reconstruction(model, num_x, num_blocks, n_train, n_test, g, device):
    """Fit a linear map residual[layer] -> the TRUE target sat(x, c), for
    every residual-stream layer 0..num_blocks, with c ~ U[1,2] (the training
    distribution, not pinned pairs).

    The target is the true label, not the model's own output: the model's
    output is exactly `residual[num_blocks] @ W_U`, so regressing onto it
    scores a trivial R²=1 at the last layer no matter how badly the model
    solves the task (see PR #223). Against the true target the last layer
    instead tops out near the model's own task fidelity.

    Always fit and evaluate noise-free (unlike the binary probe metrics
    above, which do inject eval-time noise): this analysis asks whether c's
    contribution to y is linearly decodable at a given layer, which OLS
    answers on its own (a genuinely nonlinear map just gets a low R²). Eval
    noise the fit never saw at train time instead measures how much a
    layer's fitted coefficients amplify an unrelated perturbation -- a
    layer whose fit happens to need large coefficients (e.g. because its
    encoding is scaled down) can swing to wildly negative R² for reasons
    having nothing to do with linearity."""
    layers = list(range(0, num_blocks + 1))

    def _sample(n):
        x_full, y = sample_batch(n, num_x, generator=g, device=device)
        _, caches = model.forward(x_full, return_cache=True, generator=g)
        return {lyr: caches[lyr].cpu().numpy() for lyr in layers}, y.cpu().numpy()

    train_caches, y_train = _sample(n_train)
    test_caches, y_test = _sample(n_test)

    r2 = {}
    for lyr in tqdm(layers, desc="linear-y per layer", leave=False):
        reg = LinearRegression().fit(train_caches[lyr], y_train)
        pred_te = reg.predict(test_caches[lyr])
        r2[lyr] = float(r2_score(y_test, pred_te))
    return r2


def _probe_metrics_one_featureset(
    r_lo_tr: torch.Tensor,
    r_hi_tr: torch.Tensor,
    r_lo_te: torch.Tensor,
    r_hi_te: torch.Tensor,
    probe_backend_name: str,
) -> tuple[dict, dict]:
    """`(metrics, plot_inputs)` for one feature set -- a single layer's
    activations, or several layers' concatenated. Contents as documented on
    `binary_probe_metrics_all_layers`, minus the per-layer keying."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from probe_backend import PROBE_EVAL_NEWTON_COLD_STEPS, build_probe_pipeline

    device = r_lo_tr.device
    n_train, n_test = r_lo_tr.shape[0], r_lo_te.shape[0]

    X_tr = np.concatenate([r_lo_tr.cpu().numpy(), r_hi_tr.cpu().numpy()], axis=0)
    y_tr = np.concatenate([np.zeros(n_train), np.ones(n_train)])
    X_te = np.concatenate([r_lo_te.cpu().numpy(), r_hi_te.cpu().numpy()], axis=0)
    y_te = np.concatenate([np.zeros(n_test), np.ones(n_test)])
    lda = LinearDiscriminantAnalysis().fit(X_tr, y_tr)

    X_tr_t = torch.cat([r_lo_tr, r_hi_tr], dim=0).to(PROBE_EVAL_FIT_DTYPE)
    y_tr_t = torch.cat(
        [
            torch.zeros(n_train, dtype=torch.bool, device=device),
            torch.ones(n_train, dtype=torch.bool, device=device),
        ]
    )
    pipeline = build_probe_pipeline(
        C=PROBE_C,
        max_iter=2000,
        backend=probe_backend_name,
        newton_cold_steps=PROBE_EVAL_NEWTON_COLD_STEPS,
    )
    pipeline.fit(X_tr_t, y_tr_t)
    w_probe_t, b_probe_t = pipeline.get_affine(device)
    w_probe = w_probe_t.cpu().numpy()
    b_probe = float(b_probe_t.cpu())

    mu_lo = r_lo_tr.mean(dim=0)
    mu_hi = r_hi_tr.mean(dim=0)
    w_dom = (mu_hi - mu_lo).cpu().numpy()
    midpoint = float(((mu_hi + mu_lo) / 2).cpu().numpy() @ w_dom)
    dom_boundary = LinearBoundary(w_dom, -midpoint)
    probe_boundary = LinearBoundary(w_probe, b_probe)

    metrics = {
        "dom": boundary_accuracy(dom_boundary, X_te, y_te),
        "delta_norm": float(np.linalg.norm(w_dom)),
        "logreg": boundary_accuracy(probe_boundary, X_te, y_te),
        "lda": float(lda.score(X_te, y_te)),
    }
    plot_inputs = {
        "w_dom": w_dom,
        "midpoint": midpoint,
        "w_probe": w_probe,
        "b_probe": b_probe,
        "X_te": X_te,
        "y_te": y_te,
        "dist_lo": raw_signed_distance(w_probe, b_probe, r_lo_te.cpu().numpy()),
        "dist_hi": raw_signed_distance(w_probe, b_probe, r_hi_te.cpu().numpy()),
    }
    return metrics, plot_inputs


def binary_probe_metrics_concat_layers(
    model,
    c_lo,
    c_hi,
    layers,
    g,
    probe_backend_name,
    *,
    n_train: int = PROBE_EVAL_N_TRAIN,
    n_test: int = PROBE_EVAL_N_TEST,
    train_noise=0.0,
    eval_noise=0.0,
) -> tuple[dict, dict]:
    """One probe fit over the concatenated activations of every layer in
    `layers` -- the feature set the training-time adversary sees (see
    train_adversarial_logreg's concat_caches_torch), so this is what "refit
    the run's own probe" means for a run penalized at several layers.

    Arguments and returns follow `binary_probe_metrics_all_layers`, except
    that `metrics` and `plot_inputs` describe the single concatenated feature
    set rather than being keyed by layer. For one layer the two agree; beyond
    that this probe sees sum(d_model) features and is strictly stronger, so
    its AUROC is not comparable with a per-layer one.
    """
    device = next(model.parameters()).device
    layers = sorted(layers)
    train_ds = binary_dataset_all_layers(
        model, model.num_x, n_train, c_lo, c_hi, layers, g, device, noise=train_noise
    )
    test_ds = binary_dataset_all_layers(
        model, model.num_x, n_test, c_lo, c_hi, layers, g, device, noise=eval_noise
    )

    def cat(ds, i):
        return torch.cat([ds[layer][i] for layer in layers], dim=-1)

    return _probe_metrics_one_featureset(
        cat(train_ds, 0),
        cat(train_ds, 1),
        cat(test_ds, 0),
        cat(test_ds, 1),
        probe_backend_name,
    )


def binary_probe_metrics_all_layers(
    model,
    c_lo,
    c_hi,
    layers,
    g,
    probe_backend_name,
    *,
    n_train: int = PROBE_EVAL_N_TRAIN,
    n_test: int = PROBE_EVAL_N_TEST,
    desc="layers",
    train_noise=0.0,
    eval_noise=0.0,
):
    """DoM / logreg / LDA accuracy for every layer in `layers`, one (c_lo,
    c_hi) pair, from a single shared forward pass per train/test set. One
    independent probe per layer -- contrast
    `binary_probe_metrics_concat_layers`, which fits a single probe across
    all of them.

    `n_train`/`n_test` default to the shared analysis sizes (config) rather
    than being set per script, so every report measures the same thing.

    Fits a fresh probe against the checkpoint, unlike `stored_probe_auroc`
    which scores the one the model was trained against.

    Pure data generation -- no plotting. Returns `(metrics, plot_inputs)`:
    `metrics` is {layer: {"dom", "delta_norm", "logreg", "lda"}}, and
    `plot_inputs` is {layer: {"w_dom", "midpoint", "w_probe", "b_probe",
    "X_te", "y_te", "dist_lo", "dist_hi"}}, everything a caller needs to
    later feed `plot_probe` or a layer-distribution plot for this (c_lo, c_hi)
    pair -- returned unconditionally since the forward pass and the fit
    already happened regardless of whether the caller wants a plot.

    `train_noise` and `eval_noise` are independent (each either a std or a
    noise tensor, see `capture_layers_dict`): the former affects the fit
    forward pass, the latter the test forward pass -- lets a caller ask
    whether a probe fit under noise generalizes differently than one fit
    clean.
    """
    from tqdm import tqdm

    num_x = model.num_x
    device = next(model.parameters()).device
    train_ds = binary_dataset_all_layers(
        model, num_x, n_train, c_lo, c_hi, layers, g, device, noise=train_noise
    )
    test_ds = binary_dataset_all_layers(
        model, num_x, n_test, c_lo, c_hi, layers, g, device, noise=eval_noise
    )

    metrics = {}
    plot_inputs = {}
    for layer in tqdm(layers, desc=desc, leave=False):
        metrics[layer], plot_inputs[layer] = _probe_metrics_one_featureset(
            *train_ds[layer], *test_ds[layer], probe_backend_name
        )
    return metrics, plot_inputs


@torch.no_grad()
def forward_steered(
    model: ResidualMLP,
    x_full: torch.Tensor,
    steer_layer: int,
    steer_vec: torch.Tensor | None,
    noise: Noise | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Manual replay of ResidualMLP.forward, injecting steer_vec into the
    residual stream at index steer_layer (0=embedding, i=after block i-1).
    `noise`/`generator` follow ResidualMLP.forward's convention (see there)."""
    if isinstance(noise, float):
        noise = model.generate_noise(x_full.shape[0], noise, generator)
    r = x_full @ model.W_E
    if steer_layer == 0 and steer_vec is not None:
        r = r + steer_vec
    for i, block in enumerate(model.blocks):
        r = r + block(r)
        if noise is not None and i + 1 < model.num_blocks:
            r = r + noise[i]
        if (i + 1) == steer_layer and steer_vec is not None:
            r = r + steer_vec
    y = r @ model.W_U
    return y[:, : model.num_x]


def save_plot(fig, plot_dir: str, filename: str, close: bool = True) -> str:
    """Save a finished figure and log its path. Closes it by default; pass
    close=False to leave it open for a later plt.show() (only the plots
    adversarial_report.main wants --show to actually pop should do this).
    Resolution comes from rcParams["savefig.dpi"]; set that in a calling
    script to override matplotlib's default."""
    path = os.path.join(plot_dir, filename)
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_probe(
    tag,
    layers,
    dom: LinearBoundary,
    probe: LinearBoundary,
    X_test: Float[np.ndarray, "n d"],
    y_test: Bool[np.ndarray, "n"],
    plot_dir,
):
    """n = test-set size (both classes concatenated), d = probed feature dim
    (sum of d_model over the probed layers). `dom` and `probe` are two affine
    decision boundaries over the same feature space -- a difference-of-means
    direction and a fit logreg/LDA probe -- compared side by side. `probe`'s
    weight/bias are raw (unstandardized): whatever scaler the probe was fit
    with already folded in, so this function is agnostic to which backend
    (sklearn/torch) produced them.

    Writes a histogram+ROC comparison ({tag}_L{layers}_probe.png). See also
    `plot_probe_pca` for a companion PCA-based comparison over the same
    inputs."""
    from sklearn.metrics import roc_auc_score, roc_curve

    proj_dom = dom.score(X_test)
    proj_logreg = probe.score(X_test)

    auroc_dom = roc_auc_score(y_test, proj_dom)
    auroc_logreg = roc_auc_score(y_test, proj_logreg)
    fpr_dom, tpr_dom, _ = roc_curve(y_test, proj_dom)
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, proj_logreg)

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0
    layer_str = "-".join(str(i) for i in layers)

    # --- figure 1: histograms + ROC ---
    fig, (ax_dom, ax_logreg, ax_roc) = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, proj, title in (
        (ax_dom, proj_dom, "DoM projection"),
        (ax_logreg, proj_logreg, "logreg decision function"),
    ):
        ax.hist(proj[lo_mask], bins=60, alpha=0.6, label="c=1")
        ax.hist(proj[hi_mask], bins=60, alpha=0.6, label="c=2")

        # Set plot limit to avoid outliers
        percentile_5, percentile_95 = np.percentile(proj, [5, 95])
        percentile_diff = percentile_95 - percentile_5
        ax.set_xlim(
            [
                percentile_5 - percentile_diff * 0.5,
                percentile_95 + percentile_diff * 0.5,
            ]
        )

        ax.axvline(0.0, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(title)
        ax.set_xlabel("projection (test set)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_roc.plot(fpr_dom, tpr_dom, label="DoM")
    ax_roc.plot(fpr_logreg, tpr_logreg, label="logreg")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"AUROC: DoM: {auroc_dom:.3f}; LogReg: {auroc_logreg:.3f}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect("equal", adjustable="box")

    fig.suptitle(f"probe separation ({tag}, layers={layer_str})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{layer_str}_probe.png")


def plot_probe_pca(
    tag,
    layers,
    dom: LinearBoundary,
    probe: LinearBoundary,
    X_test: Float[np.ndarray, "n d"],
    y_test: Bool[np.ndarray, "n"],
    plot_dir,
):
    """Companion to `plot_probe`, over the same inputs: writes a PCA
    comparison ({tag}_L{layers}_probe_pca.png) -- each probe's own direction
    projected out and PC1 of what's left plotted against a shared
    top-2-component PCA, showing whether any orthogonal variance still
    separates the classes. Split out since PCA is the expensive part and
    only wanted for a detailed report."""
    from sklearn.decomposition import PCA

    pca_xy = PCA(n_components=2).fit_transform(X_test)

    def _resid_pc1(w: Float[np.ndarray, "d"]):
        w_hat = w / np.linalg.norm(w)
        X_resid = X_test - np.outer(X_test @ w_hat, w_hat)
        pc1 = PCA(n_components=1).fit_transform(X_resid)[:, 0]
        return w_hat, pc1

    # each probe's own direction, projected out, then PC1 of what's left --
    # raw projections are in data coordinates (unlike a decision-score
    # projection, which is in each probe's own units).
    w_dom_hat, pc1_resid_dom = _resid_pc1(dom.w)
    w_logreg_hat, pc1_resid_logreg = _resid_pc1(probe.w)
    proj_dom_raw = X_test @ w_dom_hat
    proj_logreg_raw = X_test @ w_logreg_hat
    dom_raw_threshold = float(-dom.b / np.linalg.norm(dom.w))
    logreg_raw_threshold = float(-probe.b / np.linalg.norm(probe.w))

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0
    layer_str = "-".join(str(i) for i in layers)

    fig, (ax_pca, ax_dom_resid, ax_logreg_resid) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_pca.scatter(pca_xy[lo_mask, 0], pca_xy[lo_mask, 1], s=4, alpha=0.4, label="c=1")
    ax_pca.scatter(pca_xy[hi_mask, 0], pca_xy[hi_mask, 1], s=4, alpha=0.4, label="c=2")
    ax_pca.set_title("PCA (top 2 components)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(True, alpha=0.3)
    ax_pca.set_aspect("equal", adjustable="datalim")

    for ax, proj_raw, pc1_resid, threshold, title, xlabel in (
        (
            ax_dom_resid,
            proj_dom_raw,
            pc1_resid_dom,
            dom_raw_threshold,
            "DoM vs residual PCA",
            "DoM projection (data coords)",
        ),
        (
            ax_logreg_resid,
            proj_logreg_raw,
            pc1_resid_logreg,
            logreg_raw_threshold,
            "logreg vs residual PCA",
            "logreg projection (data coords)",
        ),
    ):
        ax.scatter(proj_raw[lo_mask], pc1_resid[lo_mask], s=4, alpha=0.4, label="c=1")
        ax.scatter(proj_raw[hi_mask], pc1_resid[hi_mask], s=4, alpha=0.4, label="c=2")
        ax.axvline(threshold, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PC1 of orthogonal residual")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"probe separation, PCA ({tag}, layers={layer_str})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{layer_str}_probe_pca.png")
