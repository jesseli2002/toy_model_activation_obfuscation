## Miscellaneous
###
(Manual)
Rename train* to train_model*, then train_probe

###
Can you modify train.py to check if the directory given by --tag is present, and if so, to not run/clobber the previous run (if --resume isn't present)?
- In a second commit, add a --tag-force option which deletes the existing directory if present.

###
What are the implications of moving model.NUM_BLOCKS into config.py?

### Security
Help me lockdown the settings.local.json against prompt injection attacks from Github.
    - The Github MCP server is linked to a personal access token that only permits access to one public repository that I own.
    - However, potentially a malicious user could e.g. create an issue with a prompt injection? or a PR?
I'm OK with a multi-pronged approach - e.g. restrict read permissions only to some subset of actions, then restrict on Github who can e.g. make PRs, so that the MCP server can only read from trusted content.
