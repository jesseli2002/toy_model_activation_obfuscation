## Miscellaneous
I have some miscellaenous code quality improvement requests. Can you go through each of these, and


- Refactor where outputs are written to: Create runs/ directory, and each run should have a directory (corresponding to tag), under which checkpoints & logs should live.
- Jaxtyping documentation for size of tensors in various methods/functions. model.forward, task_output I've noticed already are missing them, but having them elsewhere would be good too.

- ResidualMLP hard-codes number of blocks; probably a constant should be defined for that instead to make it easy to change number of blocks.
    - This requires rethinking of the forward method. Does PyTorch have a built-in module collection class that can be used?
