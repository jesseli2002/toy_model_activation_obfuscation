## Needs human review
- What is ridge regression good for?

- Simple experiment: Just see what happens if we warmstart things.



## Features
### General work
- Sweep the following:
    - N_BLOCKS
    - lambda (proportion of loss for obfuscation) -> get a Pareto frontier


###
In train_adversarial_logreg, can you add the following features:
- Add save_every_n (as is done in train_adversarial.py), which defaults to ckpt_interval
- Include probe direction in the saved checkpoint


## Code quality

### Misc
I think train_model.py is no longer useful, since its behaviour can be replicated using train_adversarial.py with --lam=0. Can you review train_model.py and see if there's any functionality there that's not present in train_adversarial.py?

### -------------
train_probe.py has capture_layers and capture_layers_dict which basically do the same thing - can you check for callers and give me a report? If it looks reasonable to do, I'll instruct you to unify them to use capture_layers_dict.



### Tests?
What unit tests could be implemented?

## Environment
- Github MCP seems to keep making the worktree de-sync from the remote origin.
