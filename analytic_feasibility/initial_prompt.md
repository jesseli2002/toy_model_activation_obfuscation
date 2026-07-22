###
Can you find an analytic solution to the task set out in this repository? The goal is to train a residual MLP model to learn the function y = sat(x, -c, +c), where:
- the inputs to the model is a vector x (densely activated) concatenated with a scalar c (always present)
- sat is the saturation function; in other words, y = min(c, max(-c, x))
- x ranges uniformly from [-3, 3]
- c ranges uniformly from [1, 2]
- (the challenge) `c` is not recoverable using a linear probe at any of the hidden layers.


Copying from plans/high_level_plan.md (the original human-written planning doc, although some parts of it are out of date):
- Inputs x_full = [x, c] (row-vector); note that this will be of size `len(x) + 1`
- Embedding matrix W_E
    - W_E is a fixed matrix, with unit norm orthogonal rows. The easiest way to do this is to make W_E be a block matrix, concatenating an identity matrix and a zero matrix appropriately to get a `len(x)+1` by `d_model` sized matrix.
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
- More MLP blocks (NUM_BLOCKS total)
- Unembedding matrix W_U
    - W_U should be fixed equal to W_E^T

Some more details/observations:
- The MLP model width is constrained to be len(x). (Note that if the width was 2*len(x) or wider, the problem could be trivially solved in one layer)
- Because of this constraint, the model must necessarily encode `c` somehow in at least the first hidden layer,
- For this analytic analysis, you can assume the residual stream width, number of blocks, and len(x) to be arbitrarily large - although I doubt you'll need, say, hundreds of dimensions for this.
- For this analysis, a reasonable proxy is - can we ensure that the mean value of the activations is constant across c?
- analytic.py show that two trivial extremes of the problem are solvable (one end: only obfuscating c and ignoring the task, by erasing c in the first layer; other end: only solving the task, using 2 layers)
- A Sonnet model with medium thinking effort found the expression `4 * relu(x - c) - relu(3 - 2 * c - x)`, which asymmetrically nonlinearly partially encodes `c` into the first layer. The constraint is that this expression should integrate over x in [-3, 3] to a constant independent of c.
- `c` cannot be linearly probed from the learned function y(x,c). So, in some sense, as long as each MLP block can build up at least one dimension of the answer at a time, you've found a solution. This suggests one potential roadmap:
    - Find some nonlinear reversible encoding of `c` into the residual stream using the first MLP block, that satisfies that the difference-of-means (for different choices of c) is zero.
    - With each MLP block, decode c and compute y_i = sat(x_i, -c, c) for at least one dimension and store in residual stream for output
    - Repeat as many blocks as needed (up to len(x)) until you have y fully constructed.

However, I'm not ruling out that this is a fundamentally impossible problem.

Be efficient with your token usage. It may or may not be more efficient to spawn cheaper subagents to explore particular directions, and/or to use sympy/python to your advantage.

Before you start working on this problem - is there anything I can clarify?
