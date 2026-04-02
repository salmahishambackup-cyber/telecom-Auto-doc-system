"""Shared utilities for LLM providers."""
from __future__ import annotations


def _confidence_heuristic(text: str) -> float:
    """Score docstring quality based on presence of key sections.

    Checks for a summary line, 'Args:' section, and 'Returns:' section.
    Returns a score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return 0.0
    sections_found = 0
    stripped = text.strip()
    # Summary line: any non-empty first line
    lines = stripped.splitlines()
    if lines and lines[0].strip():
        sections_found += 1
    if "Args:" in stripped or "Parameters:" in stripped or "param " in stripped:
        sections_found += 1
    if "Returns:" in stripped or "return " in stripped.lower():
        sections_found += 1
    return sections_found / 3.0
