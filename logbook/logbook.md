# Project Log
Logbook of research progress - what I tried, what I expected vs what happened, plans for next steps.

## Project summary
A common hypothesis: Training against a probe just makes the model learn to obfuscate its activations. What are the simplest toy models we can build to demonstrate this behaviour, and maybe gain mechanistic insight into the geometry of obfuscated activations? If we do find something - can we justify why or why not this representation will appear naturally, even without optimization pressure? Can we justify whether the model will suffer some capabilities tax as a result?

### Target deliverable
- Blog post of some sort, detailing results
    - Main writeup on personal website
    - Cross-post to LessWrong

### Project assumptions
#### Theoretical assumptions
I think these can only be experimentally validated by scaling up the model to an LLM, but there's theoretical reasons to make these assumptions
- The learned function (or functions?) is representative of a computation a larger model would need to do
- The model architecture is representative of larger model's capabilities.
    - It should be: model architecture is subset of transformer architecture
- The general concept (prereqs to bad feature are present but benign at earlier layers; bad feature gets represented non-linearly in later layers) is a good model of how bad behaviours (and their prerequisites) get represented in larger models
- The approach we're taking is representative of what could happen if you "train against a probe" (?)
    - Maybe it doesn't need to be: Being able to show a concrete example of nonlinear feature representation under optimization pressure, would be interesting in and of itself (?)

#### Empirical assumptions
- Toy model is cheap to train and will continue to be cheap to train - runs taking <30 min on local laptop.
    - Can be tackled
- Model architecture/training method is capable of implementing obfuscation
- Training against probe results leads to obfuscation

### Stretch goals
- See if obfuscation behaviour occurs when training to minimize activation on probe, rather than directly on probe's ability to measure things
- Reproduce results by fine-tuning LLM, a la Neural Chameleons

## Open questions
### Important
- Can we train `c` to be undetectable through a linear probe? How to do this? Ideas:
    - Minimize difference of means (DoM) (directly attacking difference-of-means probe) -> optimize for minimum distance between means when c=1 vs c=2
        - Possibly fails by optimizer just learning to scale everything down...
    - Minimize ratio of DoM to standard deviation along probe direction. In other words, make the activations noisy

### Back-burner
- Does it matter if $x$ is sparsely activating instead of densely activating?
- Can we try some other functions to learn?
- When training without obfuscation pressure - can we fully understand what the probed values are doing (and how to fully causaly change the results)?

### Answered questions
- Can we detect `c` with a probe? What types of probes?
    - Yes! At least when not adversarially trained. See log for 2026-07-17, but basically any probe type will work.


## Daily log
### 2026-07-20
- Claude has pointed out that I have no LayerNorm blocks

### 2026-07-19
- Running with 4 MLP blocks, model doesn't consistently fully learn the saturation function, even with $\lambda = 0.05$, although I suppose I haven't checked that the loss terms are scaled such that $\lambda \approx 0.5$ is reasonable. See this figure for $\lambda=0.1$:
![One of the functions isn't saturating](img/2026-06-19/adv_lam0.1_scratch_curves.png)
    - Note: `d_mlp = num_x`, so if model learns at least half the functions, then `c` must be represented somehow at layer 1.


- In any case, it's easier to just stick more MLP blocks on to see if something works, then dial # of blocks back later.
- Result: Logistic regression (and Linear Discriminant Analysis) probes can still detect $c$, but training process hides it from difference-of-means probe. First figure shows that model still learns the saturation function, although it's noisy:
![](img/2026-06-19/adv_NB8_lam0.1_scratch_curves.png)

Second figure shows how probe results are (for layer 6, but all the intermediate layers show similar behaviour)
![](img/2026-06-19/adv_NB8_lam0.1_scratch_c1-2_L6_probe.png)

Basically, the model is heavily shrinking the scaling factor for magnitude for $c$ (as seen in the difference between $c=1$ and $c=2$ in logistic regression), but it's still using a clean direction for it (as seen in logistic regression results).

In theory, the model could basically do this and achieve arbitrarily low loss on mean score: Just map $c \to \epsilon c$. I don't think scaling the loss value by standard deviation in the DoM probe direction would help, since the model can basically add arbitrarily high amounts of interference (see Figure 4 from https://arxiv.org/pdf/2310.06824 but imagine it in the extreme)
![](img/2026-06-19/figure4.png)

This is an interesting result, but the model is still storing the value of $c$ linearly. Ideas for harder pressure:
- Divide by standard deviation in direction of $c$ (or variance if using distance squared)
- Use LDA (closed-form) expression against model
- Simultaneously adversarially train a logistic regression model


### 2026-07-18
- Spent some time implementing adversarial training of probe & model, but haven't investigated the results fully.
    - Adversary loss (`l_probe`, for now, and up to scaling) is take distance between means for $c=1$ and $c=2$, then square it
    - Define $\lambda$ to be fraction of total loss for probe, rather than task score

### 2026-07-17
- Leaky ReLU is implemented, but upon further thought, might not be best choice (can't use exact expression as easily)
- Comment: Since `d_mlp = 32 << d_model = 512`, it's actually impossible to achieve full utilization of the residual stream (by way of a matrix rank argument). We can decrease d_model to 256 probably.
- Inspecting num_x=1 case - model learns to put the nonlinearity outside the data range (since $x \in [-3, 3]$), then never learns to put it back.
- Using deeper model (4 blocks instead of bare minimium 2) helps; much more reliable at training correct results. Guess is that it's much more likely that at least some nonlinearity are still kept in the data range after easy wins are obtained, which can be used to implement saturation. Three separate runs all resulted in good learned functions::
![Things work (0)](img/2026-06-17/nx32_0_curves.png)
![Things work (1)](img/2026-06-17/nx32_1_curves.png)
![Things work (2)](img/2026-06-17/nx32_2_curves.png)

- Training probe is highly successful for detection. On 4 layer model, looking at each of the 3 hidden layers, using difference-of-means and logistic regression probes (layer number in title can be interpreted as after the corresponding 1-indexed MLP blocks, or before the 0-indexed MLP block):
![PCA is messy at layer 1](img/2026-06-17/nx32_L1_probe.png)
![but at layer 2](img/2026-06-17/nx32_L2_probe.png)
![and layer 3, PCA is fine](img/2026-06-17/nx32_L3_probe.png)

- Casaul steering sort of works, but is imperfect. Steering with difference-of-means works better on earlier layers than later ones, but there are some features which aren't getting steered. Probably, the model uses multiple directions for $c$, and they'd all need to be steered.
![](img/2026-06-17/nx32_L1_steer.png)
![](img/2026-06-17/nx32_L2_steer.png)
![](img/2026-06-17/nx32_L3_steer.png)


### 2026-07-16
- Spent a large chunk of the day debugging permissions/sandbox issues :(
- Claude worked out the analytic solution: `sat(x,c) = x - ReLU(x-c) + ReLU(-x-c)`
    - And wrote an analytic check
- Faced issues with training with len(x) = 32. Picture is worth a thousand words - here's the function that gets learned (plot y(x) for fixed c, for each x):
![Curves that get learned - some features learn the proper saturation functions, other just seem to fit a straight line](img/2026-06-16_nx32_curves.png)
Training dynamics show that loss isn't decreasing further:
![Training dynamics: Loss plateaus](img/2026-06-16/nx32_dynamics.png)
- Notably - training seems to work when feature count is smaller (1,2, 4), although even then, higher number of features => longer training time.
- Found bug in d_mlp config
- Things to try:
    - Use leaky ReLU. Maybe we have dead neurons?
    - Increasing depth of model. Point is to give the model more nonlinearities to use, but don't want to increase width since we need c to be accessible at middle layer
    - Check weights on 1/2/4-feature model, see if the model learned what we expect



### 2026-07-15
Key requirements for this project:
- Define a function y(x,c), where x is a generic feature vector, c is a single feature, meant to represent something we’re probing for (e.g. deception. For ease of explaination, I'll continue to use deception as the concrete example, but the logic generalizes to other things we'd like to probe for)
    - Input features are $[x, c]$ (row vector)
    - As an MVP - input feature for $c$ is intended to represent some trigger for deception which we wouldn't know to probe for a-priori, while any visible trace of $c$ afterwards is meant to represent the "I am going to engage in deception" feature
- Assuming a residual stream based architecture:
    - We should probe for $c$ in one of the hidden (middle) layers
    - The function shouldn't be computable in a single block. Otherwise, the model could trivially learn to hide $c$ by doing all the computation in the first block, and have all subsequent blocks be no-ops.
        - This can be experimentally validated by seeing if loss drops on one layer vs two layer
- Concept:
    - Use the function y = sat(x, -c, c), where sat is saturation. I.e. y = min(c, max(-c, x))
    - Use only MLP blocks.
    - This function can be represented exactly with 2 * len(x) ReLUs, with each one requiring $c$. If we use 2 MLP blocks, each with width len(x), we can probably ensure that $c$ is visible in the middle layer.
    - Use densely activating $x$; this is useful to help prevent the model compressing all computation into the first MLP block. (Not realistic for actual models, but this is probably not relevant for behaviour?)
    - Stick to ReLU (or maybe leaky ReLU); that way at least in theory we can represent the learned function exactly.



- Wrote up a plan and let an agent implement it overnight.
