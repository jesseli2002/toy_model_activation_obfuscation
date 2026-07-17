# Project Log
Logbook of research progress - what I tried, what I expected vs what happened, plans for next steps.

## Project summary
A common hypothesis: Training against a probe just makes the model learn to obfuscate its activations. What are the simplest toy models we can build to demonstrate this behaviour, and maybe gain mechanistic insight into the geometry of obfuscated activations? If we do find something - can we justify why or why not this representation will appear naturally, even without optimization pressure?

## Open questions
### Important
- Can we detect `c` with a probe? What types of probes?
- Can we train `c` to be undetectable through a linear probe (e.g. difference of means -> optimize for minimum distance between means when c=1 vs c=2)

### Back-burner
- Does it matter if $x$ is sparsely activating instead of densely activating?
- Can we try some other functions to learn?

## Daily log
### 2026-07-17
- Leaky ReLU is implemented, but upon further thought, might not be best choice (can't use exact expression as easily)
- Comment: Since `d_mlp = 32 << d_model = 512`, it's actually impossible to achieve full utilization of the residual stream (by way of a matrix rank argument). We can decrease d_model to 256 probably.
- Inspecting num_x=1 case - model learns to put the nonlinearity outside the data range (since $x \in [-3, 3]$), then never learns to put it back.
- Using deeper model (4 blocks instead of bare minimium 2) helps; much more reliable at training correct results. Guess is that it's much more likely that at least some nonlinearity are still kept in the data range after easy wins are obtained, which can be used to implement saturation. Three separate runs:

Next steps: Train probe




### 2026-07-16
- Spent a large chunk of the day debugging permissions/sandbox issues :(
- Claude worked out the analytic solution: `sat(x,c) = x - ReLU(x-c) + ReLU(-x-c)`
    - And wrote an analytic check
- Faced issues with training with len(x) = 32. Picture is worth a thousand words - here's the function that gets learned (plot y(x) for fixed c, for each x):
![Curves that get learned - some features learn the proper saturation functions, other just seem to fit a straight line](img/2026-06-16_nx32_curves.png)
Training dynamics show that loss isn't decreasing further:
![Training dynamics: Loss plateaus](2026-06-16_img/nx32_dynamics.png)
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
