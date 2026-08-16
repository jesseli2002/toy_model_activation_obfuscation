# Toy Model of Activation Obfuscation

Training toy models to obfuscate their activations against linear probes.

[See the writeup here](https://jesseli2002.github.io/blog/projects/toy-model-of-activation-obfuscation/).

CLAUDE.md contains details about how this repo is structured.

Note: A curated set of run results (including the data used to generate all the plots) can be downloaded from HuggingFace Hub with the command:
```
hf download cooleytukey/toy_model_of_activation_obfuscation --repo-type model --local-dir . --include "runs/*"
```

### Miscellaneous additional training details
Here, I describe some implementation decisions I made around training settings which didn't make it to the writeup, roughly in reverse chronological order so that the newest information comes first.

- I found training probes on later layers to be more challenging than on early layers (layer 2), especially for higher values of $\lambda$. In the process, I made these training hyperparameter changes:
    - A preliminary investigation suggests that there might be dying GeLUs; my intuition is that $\lambda$ encourages the model to just turn off everything (making it challenging to later actually leran the task)
    - As a result, I found it useful to warm up the value of $\lambda$ through the training process.
    - For sweep18/sweep19, I consistently used 220k iterations, with the first 20k iterations setting $\lambda=0$.
        - Contrast the initial experiments only on layer 2, where for this model size I used only 100k iterations.
    - I found for final $\lambda=0.01$, I could immediately apply that value of lambda at iter 20k. But for $\lambda=0.1$, linear ramp-up was necessary.
        - Ramping up over 20k iters or less didn't help. Ramping 100k iters was only partially effective. So, the ramp up happens over the entire remaining 200k iterations.
        - From this experience I decided to use the same $\lambda$ warmup schedule for $\lambda=0.032$, but I didn't test the warmup schedule.
    - I found that refitting the probe on every training iteration (as opposed to every 2 iterations) reduced the final probe AUROC, with little to no penalty in success rate.
    - I found that that using residual stream noise of 0.01 (instead of 0.03 for earlier runs) was helpful. I didn't try reducing the noise further.
    - I found an exponential decay learning schedule instead of cosine decay was more effective. I tried a final learning rate fraction (lr_min_frac) of 0.05, 0.1, and 0.2, but couldn't conclusively say one was better than the other so I stuck with 0.1.
    - After making these changes, probing on multiple layers didn't require more changes.
- For training larger models: I only tried increasing model width, keeping d_mlp, d_model, and num_x in sync.
    - I reduced the learning rate from 0.003 to 0.0015. I found this to be faster than a learning rate of 0.001.
    - I found it helpful to scale number of iterations with width of model.
- I struggled with training stability. There's still some stability issues, but it's much better than before. Some of the fixes I used/tried:
    - Using a learning rate decay schedule
    - Using StableAdamW instead of AdamW
    - Lowering the default values of $\beta_1, \beta_2$ in Adam (which is justifiable because we know the learned task exactly, so there's no noise in the training data itself)
    - Using gradient clipping (per block)
    - I tried adding a detector for when loss explodes and clipping the gradient more aggressively if loss explosion was detected, but I found that to be ineffective and ultimately disabled that feature for all the runs in the final writeup.
- Training stability is not much of a problem when $\lambda=0$.

Incidentally, all of these random tuning experiments explains why the curated `runs/` list do not have sweep names in sequential order.
