"""Throwaway: compare training-loss trajectories across 2+ groups of runs,
where each group is a set of same-settings runs differing only by trial/seed
suffix `tr[N]`. All trials within a group share one color; individual trial
lines are drawn transparent, and the per-iteration median trajectory is
drawn opaque on top so it stands out.

Adapted from sweep_inspect_training.py. Settings live in constants below,
same rationale (shape still churning, argparse would be premature). Prints
to console / plt.show() only; nothing written to disk.
"""

# Each group: name -> list of run tags. Vary only the tr[N] suffix within a
# group; everything else about the group's runs should match.
# fmt: off
GROUPS = {
    # # sweep16: probe on different layers
    # "train": [f"sweep16_layer10_lam0.01_iter200k_tr{i}" for i in range(4)],

    # # sweep model size
    # "nx32_lam0.01": [f"sweep7_lam0.01_tr{i}" for i in range(15)],
    # "nx32_lam0.1": [f"sweep7_lam0.1_tr{i}" for i in range(15)],
    "nx32": [f"sweep17_lr0.0015_iter100k_lam0.01_tr{i}" for i in range(10)],
    "nx64": [f"sweep11_lr0.0015_iter200k_lam0.01_tr{i}" for i in range(10)],
    "nx128": [f"sweep14_lr0.0015_iter400k_lam0.01_tr{i}" for i in range(10)],
}

# LAMBD = "0.01"
# GROUPS = {
#     f"lam{lam}/warm{warm}": [
#         f"sweep15_layer10_lam{lam}_warm{warm}{warm}_lr0.0015_tr{i}" for i in range(4)
#     ]
#     for lam, warm in [(f"{LAMBD}", "10k"), (f"{LAMBD}", "20k")]
# } | {
#     f"lam{LAMBD}/warm0": [f"sweep13_layer10_lam{LAMBD}_tr{i}" for i in range(10)],
# }

# fmt: on
# Smoke-tested against the group above (17 trials each) -- swap in your own.
LOSS_LOWPASS_WINDOW = 1  # running mean of loss over the past this-many iters
LOSS_TYPE = "task"
TRIAL_ALPHA = 0.4
MEDIAN_LINEWIDTH = 2.5

import matplotlib.pyplot as plt

from sweep_lib.history import causal_lowpass, load_history, median_trajectory


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors

    for (group_name, tags), color in zip(GROUPS.items(), colors):
        trials = []
        for tag in tags:
            iters, losses = load_history(tag, LOSS_TYPE)
            filtered = causal_lowpass(iters, losses, LOSS_LOWPASS_WINDOW)
            trials.append((iters, filtered))
            ax.plot(iters, filtered, color=color, alpha=TRIAL_ALPHA, lw=1)

        median_iters, median_vals = median_trajectory(trials)
        ax.plot(
            median_iters,
            median_vals,
            color=color,
            alpha=1.0,
            lw=MEDIAN_LINEWIDTH,
            label=f"{group_name} (median, n={len(tags)})",
        )

    ax.set_xlabel("iter")
    ax.set_ylabel(f"{LOSS_TYPE} loss (smoothed over past {LOSS_LOWPASS_WINDOW} iters)")
    ax.set_yscale("log")
    ax.set_title("training loss by group")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
