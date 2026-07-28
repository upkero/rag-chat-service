"""Skills: one unit of model work per module.

A skill is a *job* — summarize this dialogue, extract these fields, classify this
intent. It owns a prompt from `prompts/`, calls the model through the `LLMClient`
port, and returns a `contracts/` object. It never returns the model's raw string:
the moment a caller receives free text, parsing it becomes that caller's problem and
the same brittle `json.loads` appears in three services.

One file per job, named after the job (`summarize_dialogue.py`, not `skills.py`).
A single module holding one constant is what this directory exists to prevent —
that shape promises a concept and delivers a string.
"""
