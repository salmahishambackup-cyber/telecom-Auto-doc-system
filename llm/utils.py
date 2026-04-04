"""Shared utilities for LLM providers."""
from __future__ import annotations

import re
import textwrap


def _clean_docstring(text: str) -> str:
    """Clean an LLM response so that only the docstring text remains.

    Handles common failure modes of code-generating models:

    * Strips markdown code fences (````` ``` ````` blocks), including
      multiple consecutive or unclosed fences.
    * Strips any residual inline backtick fence markers (``` ``` ```language).
    * Replaces escaped triple-quote sequences so downstream stripping works.
    * Rejects degenerate responses (backtick spam, empty text).
    * If the remaining text contains function/class definitions, extracts
      the first docstring found between triple-quotes.
    * Strips a leading ``def``/``class`` wrapper line and dedents.
    * Strips a function-signature echo (``name(\\nparams\\n) -> T:\\n``).
    * Removes duplicated content (when the model repeats the same answer).
    * Strips surrounding triple-quotes.
    * Returns empty string for any finally-degenerate result.
    """
    if not text:
        return text

    cleaned = text.strip()

    # 1. Strip markdown code fences (```python ... ``` or ``` ... ```),
    #    including multiple consecutive opening fences and unclosed blocks.
    cleaned = _strip_code_fences(cleaned)

    # 2. Remove any remaining inline backtick fence markers (safety net).
    cleaned = _strip_inline_backtick_fences(cleaned)

    # 3. Replace escaped quote sequences so later stripping can match them.
    cleaned = _strip_escaped_quotes(cleaned)

    # 4. Reject degenerate output (e.g. backtick-spam or whitespace-only).
    if _is_degenerate(cleaned):
        return ""

    # 5. If the text looks like code (contains def/class definitions),
    #    try to extract the first docstring from it.
    if _looks_like_code(cleaned):
        extracted = _extract_first_docstring(cleaned)
        if extracted:
            cleaned = extracted

    # 6. Strip a leading def/class wrapper line (Issue 1).
    cleaned = _strip_wrapping_definition(cleaned)

    # 7. Strip a function-signature echo without 'def' keyword (Issue 5).
    cleaned = _strip_signature_echo(cleaned)

    # 8. Remove duplicate content (model sometimes repeats the answer).
    cleaned = _deduplicate(cleaned)

    # 9. Strip surrounding triple-quotes if the model included them.
    cleaned = _strip_triple_quotes(cleaned)

    # 10. Final degenerate check after all cleaning.
    if _is_degenerate(cleaned):
        return ""

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences, returning the inner content.

    Handles:
    - Standard single fence pair (``` ... ```)
    - Multiple consecutive opening fences (no inner content between them)
    - Unclosed fences (treat remainder as content)
    - Nested fences
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()
    inner: list[str] = []
    depth = 0
    for line in lines:
        if line.startswith("```"):
            if depth == 0:
                # Opening fence — skip it and start collecting
                depth = 1
            else:
                # Closing fence — stop collecting
                depth = 0
        elif depth > 0:
            inner.append(line)

    result = "\n".join(inner).strip()
    # If we got nothing (e.g. multiple consecutive opening fences with no
    # content between them), fall back to stripping all fence lines.
    if not result:
        non_fence_lines = [ln for ln in lines if not ln.startswith("```")]
        result = "\n".join(non_fence_lines).strip()
    return result if result else text


_INLINE_FENCE_RE = re.compile(r"```\w*")


def _strip_inline_backtick_fences(text: str) -> str:
    """Remove any remaining ``` or ```language markers from text."""
    return _INLINE_FENCE_RE.sub("", text).strip()


def _strip_escaped_quotes(text: str) -> str:
    """Replace escaped triple-quote sequences with their literal forms."""
    return text.replace(r'\"', '"').replace(r"\'", "'")


def _is_degenerate(text: str) -> bool:
    """Return True if *text* is degenerate (empty, whitespace, or backtick spam).

    A degenerate response is one where there is no meaningful docstring
    content — for example a string that consists solely of backtick
    characters and spaces, or is empty after stripping.
    """
    stripped = text.strip()
    if not stripped:
        return True
    # Check whether all non-whitespace characters are backticks
    non_ws = stripped.replace(" ", "").replace("\n", "").replace("\t", "")
    if non_ws and all(c == "`" for c in non_ws):
        return True
    return False


_DEF_WRAPPER_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+\w+[^:]*:\s*\n",
    re.MULTILINE,
)


def _strip_wrapping_definition(text: str) -> str:
    """Remove a leading ``def``/``class`` line if the remainder is a docstring.

    Handles the failure mode where the LLM returned the full function
    signature followed by the docstring body, e.g.::

        def foo(x, y):
            \"\"\"Do something.\"\"\"

    When detected, the inner docstring text is returned.  If no
    triple-quoted block is found after the definition line, the definition
    line is stripped and the remainder is dedented.
    """
    m = _DEF_WRAPPER_RE.match(text)
    if not m:
        return text
    remainder = text[m.end():]
    extracted = _extract_first_docstring(remainder)
    if extracted:
        return extracted
    return textwrap.dedent(remainder).strip()


# Detect a multi-line function signature echo:
#   ``name(\n  params\n) -> ReturnType:\n``
_SIGNATURE_ECHO_RE = re.compile(
    r"^\s*\w+\s*\(\s*\n"   # "func_name(\n"
    r"(?:.*\n)*?"            # parameter lines (non-greedy)
    r"\s*\)[^)]*:\s*\n",    # ") -> Type:\n" or "):\n"
    re.MULTILINE,
)

# Detect a single-line function signature echo where the *entire* text is
# just ``name(params)`` or ``name(params) -> ReturnType`` with nothing else.
# Examples that should match:
#   evaluate(subset, whitelist, num_cols, cat_cols, logger: PipelineLogger)
#   foo(x: int, y: str) -> bool
# Examples that should NOT match:
#   Processes data (with optional filtering).
#   Calls evaluate(x) and returns result.
_SINGLE_LINE_SIGNATURE_RE = re.compile(
    r"^\s*\w+\s*\("           # identifier + opening paren
    r"[^)]*"                   # param text (anything except closing paren)
    r"\)"                      # closing paren
    r"(?:\s*->[^\n]*)?"        # optional return type annotation (single line)
    r"\s*:?\s*$",              # optional trailing colon + whitespace
)


def _strip_signature_echo(text: str) -> str:
    """Remove a function-signature echo that precedes the actual docstring.

    Handles **multi-line** echoes::

        transform_features(
            pl_df: pd.DataFrame,
            wl_df: pd.DataFrame,
        ) -> Tuple[csr_matrix, np.ndarray]:
            \"\"\"
            Transforms the features...

    and **single-line** echoes that constitute the entire response::

        evaluate(subset, whitelist, num_cols, cat_cols, logger: PipelineLogger)

    Multi-line: the actual docstring content is extracted from the remainder.
    Single-line: returns empty string (the entire text is the echo).
    """
    # Multi-line signature echo (with docstring body afterwards)
    m = _SIGNATURE_ECHO_RE.match(text)
    if m:
        remainder = text[m.end():]
        extracted = _extract_first_docstring(remainder)
        if extracted:
            return extracted
        return textwrap.dedent(remainder).strip()

    # Single-line signature echo (the entire text is just the signature)
    stripped = text.strip()
    if _SINGLE_LINE_SIGNATURE_RE.match(stripped):
        return ""

    return text


def _looks_like_code(text: str) -> bool:
    """Return True if *text* appears to contain Python code, not just prose."""
    # A single leading def/class is enough to flag as a code wrapper.
    if re.match(r"^\s*(?:async\s+)?(?:def|class)\s+\w+", text):
        return True

    # Multi-line signature echo without 'def': identifier( at start, with typed params
    if _SIGNATURE_ECHO_RE.match(text):
        return True

    # Single-line signature echo (entire text is just ``name(params)``)
    if _SINGLE_LINE_SIGNATURE_RE.match(text.strip()):
        return True

    code_indicators = [
        re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
        re.compile(r"^\s*class\s+\w+[\s(:]", re.MULTILINE),
        re.compile(r"^\s*self\.\w+\s*=", re.MULTILINE),
        re.compile(r"^\s*import\s+\w+", re.MULTILINE),
        re.compile(r"^\s*from\s+\w+\s+import\s+", re.MULTILINE),
    ]
    matches = sum(1 for pat in code_indicators if pat.search(text))
    return matches >= 2


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
    """Remove surrounding triple-quotes if present (raw or escaped)."""
    stripped = text.strip()
    for q in ('"""', "'''"):
        if stripped.startswith(q) and stripped.endswith(q) and len(stripped) > 6:
            inner = stripped[3:-3].strip()
            if inner:
                return inner
    return text


def _confidence_heuristic(text: str) -> float:
    """Score docstring quality based on presence of key sections.

    Checks for a summary line, 'Args:' section, and 'Returns:' section.
    Penalises responses that contain code, residual fence markers, or
    signature echoes instead of prose.
    Returns a score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return 0.0

    stripped = text.strip()

    # Degenerate output scores zero.
    if _is_degenerate(stripped):
        return 0.0

    # Penalise responses that still contain backtick fence markers.
    if _INLINE_FENCE_RE.search(stripped):
        return 0.1

    # Penalise responses that start with a multi-line signature echo.
    if _SIGNATURE_ECHO_RE.match(stripped):
        return 0.1

    # Penalise responses that are a single-line signature echo.
    if _SINGLE_LINE_SIGNATURE_RE.match(stripped):
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
