"""Shared utilities for LLM providers."""
from __future__ import annotations

import re


def _clean_docstring(text: str) -> str:
    """Clean an LLM response so that only the docstring text remains.

    Handles common failure modes of code-generating models:

    * Strips markdown code fences (````` ``` ````` blocks).
    * Unescapes escaped quote sequences (``\\\"`` → ``"``).
    * Rejects degenerate / garbage output (backtick spam, token repetition).
    * If the remaining text contains function/class definitions, extracts
      the first docstring found between triple-quotes.
    * Strips a bare leading ``def``/``class`` wrapper around the docstring.
    * Removes duplicated content (when the model repeats the same answer).
    * Strips leading/trailing whitespace.
    """
    if not text:
        return text

    cleaned = text.strip()

    # 1. Strip markdown code fences (```python ... ``` or ``` ... ```)
    cleaned = _strip_code_fences(cleaned)

    # 2. Unescape escaped quote sequences so later steps can match properly.
    cleaned = _strip_escaped_quotes(cleaned)

    # 3. Early degenerate check — reject garbage before further processing.
    if _is_degenerate(cleaned):
        return ""

    # 4. If the text looks like code (contains def/class definitions),
    #    try to extract the first docstring from it.
    if _looks_like_code(cleaned):
        extracted = _extract_first_docstring(cleaned)
        if extracted:
            cleaned = extracted

    # 5. Strip a leading function/class definition wrapping the docstring.
    cleaned = _strip_wrapping_definition(cleaned)

    # 6. Remove duplicate content (model sometimes repeats the answer).
    cleaned = _deduplicate(cleaned)

    # 7. Strip surrounding triple-quotes if the model included them.
    cleaned = _strip_triple_quotes(cleaned)

    # 8. Final degenerate check — catch anything that survived the pipeline.
    if _is_degenerate(cleaned):
        return ""

    return cleaned.strip()


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences, returning the inner content."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()
    inner: list[str] = []
    in_block = False
    for line in lines:
        if line.startswith("```") and not in_block:
            in_block = True
            continue
        if line.startswith("```") and in_block:
            in_block = False
            continue
        if in_block:
            inner.append(line)

    result = "\n".join(inner).strip()
    return result if result else text


_MIN_CODE_INDICATORS = 2

_LEADING_DEF_OR_CLASS_RE = re.compile(r"^\s*(?:def|class)\s+\w+", re.MULTILINE)


def _looks_like_code(text: str) -> bool:
    """Return True if *text* appears to contain Python code, not just prose.

    A single ``def``/``class`` at the very start of the text is enough to
    flag it as code (common LLM failure mode: wrapping the docstring inside
    the function definition).  Otherwise at least two code indicators are
    required.
    """
    # A leading def/class is a strong single-indicator — treat it as code.
    if _LEADING_DEF_OR_CLASS_RE.match(text.lstrip()):
        return True

    code_indicators = [
        re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
        re.compile(r"^\s*class\s+\w+[\s(:]", re.MULTILINE),
        re.compile(r"^\s*self\.\w+\s*=", re.MULTILINE),
        re.compile(r"^\s*import\s+\w+", re.MULTILINE),
        re.compile(r"^\s*from\s+\w+\s+import\s+", re.MULTILINE),
    ]
    matches = sum(1 for pat in code_indicators if pat.search(text))
    return matches >= _MIN_CODE_INDICATORS


_TRIPLE_QUOTE_RE = re.compile(
    r'(?:\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')', re.DOTALL
)


def _extract_first_docstring(text: str) -> str:
    """Extract the first triple-quoted docstring from *text*.

    Returns the inner text of the first ``\"\"\"...\"\"\"`` or
    ``'''...'''`` block found, or an empty string if none is found.
    """
    m = _TRIPLE_QUOTE_RE.search(text)
    if m:
        return (m.group(1) or m.group(2) or "").strip()
    return ""


def _deduplicate(text: str) -> str:
    """Remove duplicated content from the response.

    Some models repeat the entire answer.  If the text can be split into
    two (nearly) identical halves, return just the first one.
    """
    stripped = text.strip()
    length = len(stripped)
    if length < 40:
        return text

    # Try splitting at the midpoint (± 20 chars) and check for repetition.
    mid = length // 2
    for offset in range(0, min(20, mid)):
        for pos in (mid + offset, mid - offset):
            if pos <= 0 or pos >= length:
                continue
            first_half = stripped[:pos].strip()
            second_half = stripped[pos:].strip()
            if first_half and second_half and first_half == second_half:
                return first_half

    return text


def _strip_triple_quotes(text: str) -> str:
    """Remove surrounding triple-quotes if present."""
    stripped = text.strip()
    for q in ('"""', "'''"):
        if stripped.startswith(q) and stripped.endswith(q) and len(stripped) > 6:
            return stripped[3:-3].strip()
    return text


def _strip_escaped_quotes(text: str) -> str:
    """Unescape escaped quote sequences left by the LLM.

    Replaces ``\\"`` → ``"`` and ``\\'`` → ``'`` so that downstream steps
    (e.g. :func:`_strip_triple_quotes` and :func:`_extract_first_docstring`)
    can match the resulting real quote characters.
    """
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    return text


_WRAPPING_DEF_RE = re.compile(
    r'^\s*(?:def|class)\s+\w+[^\n]*\n'  # leading def/class line
    r'\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')',  # immediately followed by docstring
    re.DOTALL,
)


def _strip_wrapping_definition(text: str) -> str:
    """Remove a leading ``def``/``class`` line wrapping the docstring body.

    When the LLM returns the full function signature plus the triple-quoted
    docstring instead of just the docstring text, this function extracts the
    inner docstring content.  Returns *text* unchanged if the pattern is not
    found.
    """
    m = _WRAPPING_DEF_RE.match(text.strip())
    if m:
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if inner is not None:
            return inner.strip()
    return text


_BACKTICK_SPAM_RE = re.compile(r"^[\s`]+$")
_WHITESPACE_TOKEN_RE = re.compile(r"\S+")


def _is_degenerate(text: str) -> bool:
    """Return True if *text* is degenerate / garbage LLM output.

    Detects three patterns:

    1. Pure backtick / whitespace spam (e.g. ````` ``` ``` ...````).
    2. A stream of only one unique non-whitespace token (``the the the ...``).
    3. Any text where more than 85% of tokens are the same token.
    """
    stripped = text.strip()
    if not stripped:
        return False

    # 1. Pure backtick/whitespace spam.
    if _BACKTICK_SPAM_RE.match(stripped):
        return True

    tokens = _WHITESPACE_TOKEN_RE.findall(stripped)
    if not tokens:
        return False

    # 2 & 3. Single-token or >85% dominant-token repetition.
    if len(tokens) >= 4:
        most_common_count = max(tokens.count(t) for t in set(tokens))
        if most_common_count / len(tokens) > 0.85:
            return True

    return False


def _confidence_heuristic(text: str) -> float:
    """Score docstring quality based on presence of key sections.

    Checks for a summary line, 'Args:' section, and 'Returns:' section.
    Penalises responses that contain code instead of prose or are degenerate.
    Returns a score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return 0.0

    stripped = text.strip()

    # Degenerate / garbage output scores zero.
    if _is_degenerate(stripped):
        return 0.0

    # Penalise responses that look like code rather than a docstring.
    if _looks_like_code(stripped):
        return 0.1

    sections_found = 0
    # Summary line: any non-empty first line
    lines = stripped.splitlines()
    if lines and lines[0].strip():
        sections_found += 1
    if "Args:" in stripped or "Parameters:" in stripped or "param " in stripped:
        sections_found += 1
    if "Returns:" in stripped or "return " in stripped.lower():
        sections_found += 1
    return sections_found / 3.0
