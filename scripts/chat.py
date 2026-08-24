#!/usr/bin/env python3
"""Tiny smoke test: send one prompt to gemma3:4b via Ollama and print the reply.

Usage:
    python scripts/chat.py "What is lateral movement?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.clients import OllamaClient  # noqa: E402
from gemma_cyber.evaluation.harness import BASELINE_SYSTEM_PROMPT  # noqa: E402


def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "In one sentence, what is lateral movement?"
    client = OllamaClient()
    if not client.is_available():
        print("ERROR: Ollama not reachable at http://localhost:11434", file=sys.stderr)
        return 2
    result = client.generate(prompt, system=BASELINE_SYSTEM_PROMPT)
    print(result.text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
