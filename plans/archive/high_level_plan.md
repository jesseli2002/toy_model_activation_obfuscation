There's a hypothesized concern in AI safety, that training against a (linear) probe will lead to the model obfuscating its activations. I want to build a toy model of this, similar to how "Toy Models of Superposition" built simple one-hidden-layer models to demonstrate superposition in the presence of sparsity, or how the Compressed Computation model (introduced in https://arxiv.org/pdf/2501.14926 ) built a simple model to represent computation in superposition. (I'm aware that there's criticism of the Compressed Computation model and whether it actually represents computation in superposition, e.g. https://www.lesswrong.com/posts/ZxFchCFJFcgysYsT9/compressed-computation-is-probably-not-computation-in, but that's not relevant here.)

Just to explore the problem a bit, let's start with the first idea I have. The model should learn the function y = sat(x, -c, +c), where:
- the inputs to the model is a vector x (densely activated) concatenated with a scalar c (always present)
- sat is the saturation function; in other words, y = min(c, max(-c, x))
- x ranges uniformly from [-3, 3]
- c ranges uniformly from [1, 2]
We'll later probe for c and see if it's linearly represented in the internals. The model architecture is a 2-layer residual MLP, with residual stream size of `d_model`. x is length `num_x`, and the MLP layers have width `d_mlp`. In order, we'll have the following blocks:

- Inputs x_full = [x, c] (row-vector); note that this will be of size `num_x + 1`
- Embedding matrix W_E
    - W_E is a fixed matrix, with unit norm orthogonal rows. The easiest way to do this is to make W_E be a block matrix, concatenating an identity matrix and a zero matrix appropriately to get a `num_x+1` by `d_model` sized matrix.
    - The initial residual r_0 = x_full @ W_E
- MLP block 0, consisting of:
    - W_in_0 input matrix and b_in_0 bias term
    - ReLU
    - W_out_0 output matrix and b_out_0 bias term
    - so that the output of the block is o_0 = ReLU(r_0 @ W_in_0 + b_in_0) @ W_out_0 + b_out_0
    - Added back to residual, we have r_1 = r_0 + o_0
- MLP block 1, same design as MLP block 0
    - output of the block is o_1 = ReLU(r_1 @ W_in_1 + b_in_1) @ W_out_1 + b_out_1
    - Residual r_2 = r_1 + o_1
- Unembedding matrix W_U
    - W_U should be fixed equal to W_E^T

I suspect (but haven't rigorously proven) that by setting `d_mlp = num_x + 1`, the model should be able to learn the function `y` exactly, while preserving the value `c` in the residual stream.

As a greenfield project, there's three initial steps we should take, gated by some checks. If a gate check fails, pause development and inform me (the user), so we can investigate further. Generate a plan for each step and associated gate (so 3 plans total), then task a fresh Sonnet subagent (one new agent for each plan) with implementing each of the plans sequentially (or stopping early, if a gate check fails). (To be clear - because the steps are sequential, realistically there will only be one subagent running at a time.) If it helps, you can generate a plan for yourself as well, to keep you on track for the whole project.

## Step 1: Train model architecture
Create a Python script train.py that implements and trains this architecture. Use PyTorch. The device should be cuda if available, otherwise cpu. Train with float32.

Hyperparameters like residual stream size should be defined as constants to make them easy to tune. Here are some initial values to try:
- d_model = 512
- num_x = 32
- d_mlp = num_x + 1

The input data is entirely synthetic; use random sampling to generate data on the fly.

### Important engineering features
Don't forget to include these features:
- Manual starting seed. Don't pick 0, use a random number like 913768
- Checkpointing of progress and resumability from a checkpoint
- Avoid excessive checkpointing. In particular, disk usage should stay below 8GB, although I suspect you'll need far less than that.
- A second check if loss goes below some number (some appropriate multiple of machine precision) can be used to check if the model has learned to exactly reproduce the results.
- There will be an extra output slot corresponding to where `c` was (since `y` has one dimension fewer than the input data) - that output should not be included in the loss.

### Gate 1: Confirm the model learned the function y(x,c)
- Once training has concluded, evaluate the maximum absolute element-wise estimation error for a large number of data points. This should be small.
    - You may need to periodically pause training (e.g. by temporarily hard-coding an iteration limit, then on the next run, specifing a starting checkpoing) to check the training status.
    - If loss seems to stop decreasing, and good evaluation perforance hasn't been recovered, THIS GATE CHECK FAILS.
    - Use your judgement for other reasons this gate check should fail. In particular, make sure to periodically check that things are still making sense - if they aren't, stop and ask the user for help debugging.
    - This should be done in a new script (e.g. train_validation.py)

- Create a script to load a specified checkpoint and make the following plots in the plot/ directory:
    - Training dynamics plot: Iteration count vs loss, iteration count vs maximum absolute element-wise error
    - Plot of y(x) for each of a number of fixed values of c (say, [1, 1.333, 1.667, 2]). This should be `num_x` lines in one plot, showing that they all learned the `sat` function.
    - This should be done in a new script (e.g. train_plot.py)


## Step 2: Train probe to detect c
Using the trained model, train a linear probe to detect $c$. This is a toy problem, so it's inexpensive to try multiple probe types - I'm aware of difference-of-means and logistic regression, so you can start with those, but if there's one or two other common probe types you're aware of, feel free to add thsoe to the list.

This should be done in a new script (probe.py)

The general approach:
- Generate training data for the probe by creating one dataset of random data with $c=1$, and another dataset (same size) with $c=2$
- Run a forward pass, track activations at the hidden layer (r_1)
- Use the probe to find a direction for $c_1, c_2$

### Gate 2: Confirm the probe has detected c
For a toy problem like this, I expect c to be very cleanly identifiable. Run the following checks; if they fail, THIS GATE CHECK FAILS.
- The probe should be able to estimate the value of $c$ to reasonably good accuracy, on a fresh test dataset of probe training data (that it wasn't trained on).
    - Accuracy will be dependent on how much training data the probe has - so that's one knob you can turn to try to improve accuracy
    - However, as before - don't turn it too far. This should be a simple problem, and if things don't make sense, STOP - and ask the user for help debugging.
    - This should be done in a new script

## Step 3: Adversarially train a new model
Adversarially train the original model and a probe. For this step, just use a difference-of-means probe as in Step 2; the loss term would then be the absolute difference of means (so ideally, the model learns to have activations with c=1 and c=2 have the same mean).

- This should be done in a new script

There is no gate check for this step - if you get this far, just run one training run, then stop and we can pick through the results together.

# References
Because this is a locked-down sandbox, web access for references is extremely limited. To ameliorate this somewhat, I've downloaded some static copies of referenced literature in the references/ directory.

# Environment
This is a greenfield project. There's no git history of anything. The sandbox prevents you from accidentally deleting anything outside the working directory, so I felt it safe to give you Bash(*) access.
