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


### Adversary ideas
I have some ideas on variations to the probe loss (`l_probe`) in train_adversarial.py:
- Scale loss term by standard deviation of data in direction of difference-of-means, so that (normalized) standard deviation is 1. (If using distance squared, this should use variance instead)
    - Example of something the model could learn: Putting activations for c=1 in the middle, and c=2 on either end.
- Train on absolute distance between means, rather than distance squared
    - This can be done with/without the previous bullet
- Train on LDA discrimination ability. Requires implementing the formula for LDA in PyTorch.

In total, there should be 5 options for the loss term to be used (2x2 from the first two bullets, 1 from using LDA discrimination). My hope is that with at least one of these options, the model learns to represent `c` nonlinearly, so that it can no longer be detected by linear probes.

What pitfalls could you see coming from implementing these ideas? Any trivial solutions that the model could learn?


#### (next response)
I'm tasking an agent with implementing LayerNorm. In the meantime, some more questions:
- If combined with some scaling/normalizing behaviour, would the training on absolute distance present a meaningful difference (compared to squared distance)? My intuition says that this prevents the model from finding a solution that's "good enough" (since a small distance, squared, gets even smaller loss).
- TBH my background is pretty shaky w.r.t how LDA works, although I'm generally fine with linear algebra - can you give me a primer on how it's used in this context?
- On cross-layer smearing - is that possible, even from a theory standpoint? (For a feature to be not linearly separable at two different layers, but separable when concatenated)? Is there a small numerical example you could give for intuition?


<!--
- Scale loss term by standard deviation of data in direction of difference-of-means, so that (normalized) standard deviation is 1
    - Example of thing model could learn: Putting activations for c=1 in the middle, and c=2 on either end.
- Train on LDA discrimination ability
- Train on absolute distance between means, rather than distance squared -->


- Adversarial simultaneous training of logistic regression probe and model
- Claude mentioned in previous discussion that train_model and train_adversarial were different enough to make unifying train_model and train_adversarial challenging, without introducing some sort of loss function hook.

- Penalize lopsided variances (where some axes have much smaller variances than others)
## Miscellaneous code quality


## Environment
