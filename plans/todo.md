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

###
In the probe graph results (train_probe.plot_probe), I have some requests, somewhat related:
- In the 4th scatter graph - can the X-axis be in unscaled coordinates (i.e. data coordinates, rather than normalized by StandardScaler)?
- Can you make the PCA graphs (last two graphs) have axis equal on X and Y axis, adjustable=datalim?
- So that the last two plots have a square~ish aspect ratio - the figure needs to be made taller




### Adversary ideas
- Train on LDA discrimination?
- Train on absolute distance between means, rather than distance squared
- Claude mentioned in previous discussion that train_model and train_adversarial were different enough to make unifying train_model and train_adversarial challenging.


## Miscellaneous code quality


### Improvements to adversarial_report.py
binary_probe_metrics():
- num_x and device can be recovered from model
- tag comes from global args.tag; is there a reason to pass it in rather than just reading from global args.tag?

There might be other places in this script where a similar issue applies
