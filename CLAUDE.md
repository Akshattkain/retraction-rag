# Agent instructions

@CONTEXT.md

## Working rules
- Use .venv/bin/python for everything. Don't pip install without asking.
- No network access unless the task says so.
- Show the script and wait for my approval before running anything that writes to
  data/processed or takes more than a few seconds.
- Save each stage's printed output to results/<stage>.txt.
- Only append to CONTEXT.md; don't rewrite it.
- Don't run git commands. I commit myself.
- Never touch data/ or third_party/ contents beyond reading them.
