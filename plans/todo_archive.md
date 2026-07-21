This file holds various completed TODOs which I thought were important enough to keep a history of. Not all (or even most) todos get moved here.

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
