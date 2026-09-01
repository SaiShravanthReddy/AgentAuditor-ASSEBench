"""Shared pytest setup.

Sets dummy API credentials before any AgentAuditor module is imported, so GPTConfig() (read at
class-construction time throughout the tasks/ modules) doesn't fail or warn just from module
import. No network calls happen anywhere in this test suite - see each test file's own docstring
for what's covered and, more importantly, what isn't (nothing that requires a real LLM call, a
real embedding model, or real pipeline data).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AGENTAUDITOR_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("AGENTAUDITOR_API_BASE", "https://example.invalid/v1")

# Allow `import AgentAuditor.tasks.X` when running pytest from the repo root without an install step.
sys.path.insert(0, str(Path(__file__).parent.parent))
