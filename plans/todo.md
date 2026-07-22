## Needs human review
- What is ridge regression good for?

- Simple experiment: Just see what happens if we warmstart things.



## Features
### General work
- Sweep the following:
    - N_BLOCKS
    - lambda (proportion of loss for obfuscation) -> get a Pareto frontier

###
train_adversarial_logreg isn't robust to ctrl-c keyboardinterrupt - put training loop in a function, wrap the function with a try/except to save the final checkpoint


###
In train_adversarial_logreg.py, can you add the following features:
- Add save_every_n (as is done in train_adversarial.py), which defaults to ckpt_interval
- Include probe direction in the saved checkpoint

(Aside for human: adversarial_report may need to be forked, since train_adversarial_logreg has different behaviour - most notably, we'd like to report on probe direction.)

###
I'd like to try to speed up train_adversarial_logreg.py, by implementing logistic regression on GPU. This is a multi-step process:
- Creating an implementation of logistic regression with good control over its training process
    - It should use similar normalization & regularization. If you're uncertain how the SciPy implementation works, don't guess; ask me to help get you access to more documentation. However, if you are sure your knowledge covers it (e.g. because it's a standard technique), it's OK to just implement it
    - Create unit tests for the implementation using pytest
- Run timing comparisons between CPU vs GPU logistic regression, for sample data.
    - This requires user help, since the sandbox you're working in doesn't have GPU access. Write the script, then ask user to run it.
    - Assume dataset size of 10000 points, and vary input data dimension from 32 to 1024 logarithmically (e.g. one point for each power of two)
- Implement the GPU version in train_adversarial_logreg.py. Since agents might run that script as a smoke test, depending on whether CUDA is present at the start of the script, switch between a CPU and GPU implementation.
    - To preserve code quality, it's possible (but I'm uncertain) that you should implement an adapter for the scikit-learn LogisticRegression pipeline. Use good judgement; it might be the case that you can have the GPU implementation match the CPu impl interface. (then again, there might be real differences that make it impractical)

Take care - there's some active pull requests which haven't been merged to master, which this code will need to build upon.

Create a plan first, in particular working out what the interfaces should be to make the most sense and require the least complexity.



## Code quality

### Misc
- Results of train_adversarial_logreg.py can't be parsed successfully by adversarial_report.py - it's failing with this error:
```
Traceback (most recent call last):
  File "/home/jesse/ml/toy_probe_hiding/adversarial_report.py", line 493, in <module>
    if __name__ == "__main__":
        ^^^^^^^^^^
  File "/home/jesse/ml/toy_probe_hiding/adversarial_report.py", line 481, in main
    history = json.load(f)
  File "/home/jesse/ml/toy_probe_hiding/adversarial_report.py", line 254, in _plot_training_traces
    key = str(lyr)
KeyError: 'delta_norms'
```
Note - line numbers might be out of date. But I think the problem is that delta_norms isn't being saved to history, although I'm not certain that's meaningful for train_adversarial_logreg. Can you get to the bottom of this?



### -------------
train_probe.py has capture_layers and capture_layers_dict which basically do the same thing - can you check for callers and give me a report? If it looks reasonable to do, I'll instruct you to unify them to use capture_layers_dict.

###
Referring to PR40: there's one issue, which is that a KeyboardInterrupt could happen while the checkpoint file is being saved, which would corrupt the saved data. Can you address this problem with an appropriate context manager/signal handler, that defers the keyboard interrupt until the critical code is finished, then push commits to update the PR?



### Tests?
What unit tests could be implemented?

###
/review #40
In particular, make sure the diffs don't change anything unnecessary, and achieves the objectives/features in a simple rather than overengineered manner.
Do not change the code, only add comments.

## Environment
- Github MCP seems to keep making the worktree de-sync from the remote origin.
