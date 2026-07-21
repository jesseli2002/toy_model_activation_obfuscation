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
Another direction I'd like to explore: What would it look like to simultaneously train a model and an adversarial logistic regression probe? Would you need two simultaneous optimizers? What are the pitfalls?


###
- Adversarial simultaneous training of logistic regression probe and model
- Claude mentioned in previous discussion that train_model and train_adversarial were different enough to make unifying train_model and train_adversarial challenging, without introducing some sort of loss function hook.


###
I think train_model.py is no longer useful, since its behaviour can be replicated using train_adversarial.py with --lam=0. Can you review train_model.py and see if there's any functionality there that's not present in train_adversarial.py?


## Miscellaneous code quality
I have some miscellaneous code quality improvement tasks. I need you to review them and orchestrate some subagents to handle them. I'm not time constrained, and some of these changes will affect each other, so it's probably best to handle them one at a time; that being said, they should still each be in different branches for different PRs - so later tasks branch off of earlier branches. I don't think these tasks are particularly difficult, so subagents can use Sonnet models. Review all tasks and let me know any questions you have, before starting any code changes.

### adversarial_report.py default computations
I'm finding that in adversarial_report.py, I'm not really reading most of the console output, only looking at the generated graphs. Can you look over the code and add an optional --detailed flag, which if enabled, computes more statistics & reporting, but by default, only computes what's necessary to get the plot results?

### Public vs private functions in modules
I'm looking to improve code quality. At this point, there's a number of scripts with functions that may or may not get imported by other scripts. If I'm not mistaken, the Python convention is to prefix functions with an underscore if they're only used locally; this hasn't been followed. Can you audit the codebase and see if this change can be made?

### Defaults
I'm trying to avoid scattering default behaviour/options across the codebase, to make it easier to isolate exactly which line is responsible for a given default. To that extent, I think the best approach is to move defaults away from the CLI definition and constants in config.py, and into configuration classes - ResidualMLPConfig and AdversarialConfig. That being said, I'm not certain this is the best approach, so if you have other ideas I'd like to hear them.
- Off the top of my head, whether layer_norm is enabled, in train_adversarial.py, is de-facto defaulted by CLI rather than by ResidualMLPConfig.
- There could be other cases.
Audit train_adversarial.py and train_model.py for other default CLI options.


### -------------
Help me speed up adversarial_report.py. There's one spot I think could use improvement but I'm not sure, and there could be other easy wins.
- When training probes, it seems like it runs a forward pass for every layer, and ends up recomping the same data each time. It'd be more efficient to run the forward pass once, cache all the results, then train & test probes accordingly for each layer.

### Tests?
What unit tests could be implemented?

## Environment
- Github MCP seems to keep making the worktree de-sync from the remote origin.
