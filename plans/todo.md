## Needs human review
- On adversarially trained probe:
    - Why is LDA performing much better?
        - What happens if we do logistic regression with C=np.inf (no regularization)?
    - What is ridge regression good for?


## Miscellaneous code quality

### Improvements to adversarial_report.py
- Logistic regression data should be normalized; use StandardScaler(). This applies in train_probe.py and adversarial_report.py
    - Correct me if I'm wrong, but difference-of-means doesn't have such a need for normalization.
- Add a plot of the learned function (similar to the one in train_model_plot.plot_curves - actually, doing a little bit of refactoring so that the plot code can be imported and reused would be ideal)
- Add the histogram and PCA plots from train_probe.py. Similarly to the previous bullet - some light refactoring so that the code can be reused would be ideal.


###
Do we nead the lr_cosine decay? I'd like to validate it helps meaningfully (especially since, IIRC, Adam is adaptive and will automatically pick up an effective learning rate), and prune it if they're unnecessary.

###
There's a lot of generator constructions, repeated multiple times. I think it'd be better to create a single global generator at the start once?
