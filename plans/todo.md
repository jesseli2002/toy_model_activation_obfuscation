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


### Adversary ideas
- Train on LDA discrimination?
- Train on absolute distance between means, rather than distance squared
- Claude mentioned in previous discussion that train_model and train_adversarial were different enough to make unifying train_model and train_adversarial challenging.


## Miscellaneous code quality


## Environment
- Setup hook to automatically run black formatter
