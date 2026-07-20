## Needs human review
- On adversarially trained probe:
    - Why is LDA performing much better?
        - What happens if we do logistic regression with C=np.inf (no regularization)?
    - What is ridge regression good for?

## Features
### General work
- Sweep the following:
    - N_BLOCKS
    - lambda (proportion of loss for obfuscation) -> get a Pareto frontier


I'd like to implement LayerNorm as an optional (but on-by-default for new constructions) feature in model.py.
- Implement LayerNorm

###
Another direction I'd like to explore: What would it look like to simultaneously train a model and an adversarial logistic regression probe? Would you need two simultaneous optimizers? What are the pitfalls?

### Adversary ideas
I have some ideas on variations to the probe loss (`l_probe`) in train_adversarial.py:
- Scale loss term by standard deviation of data in direction of difference-of-means, so that (normalized) standard deviation is 1. (If using distance squared, this should use variance instead)
    - Example of something the model could learn: Putting activations for c=1 in the middle, and c=2 on either end.
- Train on absolute distance between means, rather than distance squared
    - This can be done with/without the previous bullet
- Train on LDA discrimination ability. Requires implementing the formula for LDA in PyTorch.

In total, there should be 5 options for the loss term to be used (2x2 from the first two bullets, 1 from using LDA discrimination). My hope is that with at least one of these options, the model learns to represent `c` nonlinearly, so that it can no longer be detected by linear probes.

What pitfalls could you see coming from implementing these ideas? Any trivial solutions that the model could learn?

Response from Claude:
- Biggest problem is there's a class of tricks where the model learns to just shrink the representation of $c$ while keeping it linear.
    - Normalization by std/variance should help, as does LDA, but regularizer fights back.
- LayerNorm isn't going to solve problem of shrinking. Recall LayerNorm just makes norm ~= 1.
    - LayerNorm doesn't touch residual stream, where a shrinked c could live
    - LayerNorm only closes the hack for uniform scaling.


####
Based on your feedback, I think we should do the following:
- Implement the variance/stddev-based scaling, leaving it optional but on by-default. For regularization, pick a regularization constant that's on the order of machine epsilon - so that the model can't use that exit path without facing numerical noise
- Implement the distance rather than distance squared loss function as an alternative
- Implement LDA-based scoring, with your note on using solve() instead for numerical stability.
- This would probably be a new CLI option, with 5 options, and the default one being distance^2 scaled by variance.

Is this reasonable?

Let's change the new default to LDA-based; it seems to have the best chance of finding something new. I'd like you to make an implementation plan for a Sonnet subagent. Note that in the code, there's now an option for legacy defaults to help with backward compatibility; probably that needs to be set so that the legacy loss variant is squared unnormalized.


- Adversarial simultaneous training of logistic regression probe and model
- Claude mentioned in previous discussion that train_model and train_adversarial were different enough to make unifying train_model and train_adversarial challenging, without introducing some sort of loss function hook.

- Penalize lopsided variances (where some axes have much smaller variances than others) (??)

## Miscellaneous code quality
I'm trying to avoid scattering default behaviour/options across the codebase, to make it easier to isolate exactly which line is responsible for a given default. To that extent, I need you to help with moving defaults away from the CLI and into the model config object (ResidualMLPConfig).
- Off the top of my head, whether layer_norm is enabled, in train_adversarial.py, is de-facto defaulted by CLI rather than by ResidualMLPConfig.
- There could be other cases.

Audit train_adversarial.py for other default CLI options.

### train_adversarial.py
There's gotta be a cleaner, less redundant interface than what's going on with delta_means_from_x, probe_delta_means, and probe_caches. Can you think through some alternatives that make logical sense?

### Tests?
What unit tests could be implemented?

## Environment
- Github MCP seems to keep making the worktree de-sync from the remote origin.
