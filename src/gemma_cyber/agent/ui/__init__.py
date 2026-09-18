"""Terminal presentation for `gemma4`: Rich rendering + Prompt Toolkit input.

UI code subscribes to runtime events; it never implements the agent loop and is
never imported by `tools/` or `providers/` (plan §6 dependency direction).
"""
