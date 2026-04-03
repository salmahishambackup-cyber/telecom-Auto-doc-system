"""Confidence improver agent.

Identifies weak docstring entries for regeneration and builds improved
prompts that include feedback about what went wrong.
"""
from __future__ import annotations

from models.schemas import Docstrings

_STUB_MAX_LENGTH = 20
_DEFAULT_MAX_REGENERATIONS = 50


class ConfidenceImprover:
    """Identify weak docstring entries and build improved regeneration prompts.

    Parameters
    ----------
    max_regenerations:
        Maximum number of entries returned for regeneration (default 50).
    """

    def __init__(self, max_regenerations: int = _DEFAULT_MAX_REGENERATIONS) -> None:
        self.max_regenerations = max_regenerations

    def find_weak_entries(
        self,
        docstrings: Docstrings,
        threshold: float = 0.5,
        validation_issues: list[dict] | None = None,
    ) -> list[dict]:
        """Return entries that should be regenerated.

        An entry is considered weak if any of the following apply:

        * Its confidence score is below *threshold*.
        * It appears in *validation_issues*.
        * Its docstring is shorter than 20 characters (stub).

        Parameters
        ----------
        docstrings:
            The :class:`~models.schemas.Docstrings` container to inspect.
        threshold:
            Confidence score below which an entry is considered weak.
        validation_issues:
            Optional list of issue dicts from
            :class:`~agents.docstring_validator.DocstringValidator`.
            Entries with matching ``function_id`` values are flagged
            regardless of their confidence score.

        Returns
        -------
        list[dict]
            Each dict has keys: ``function_id``, ``reason``,
            ``current_confidence``, ``current_docstring``.  Capped at
            :attr:`max_regenerations`.
        """
        flagged_ids: set[str] = set()
        if validation_issues:
            flagged_ids = {issue["function_id"] for issue in validation_issues}

        results: list[dict] = []
        seen: set[str] = set()

        for entry in docstrings.entries:
            if entry.function_id in seen:
                continue

            reasons: list[str] = []

            if entry.confidence < threshold:
                reasons.append(f"confidence {entry.confidence:.2f} below threshold {threshold:.2f}")

            if entry.function_id in flagged_ids:
                reasons.append("failed validation checks")

            docstring_text = entry.docstring or ""
            if len(docstring_text.strip()) < _STUB_MAX_LENGTH:
                reasons.append("docstring is a stub (< 20 chars)")

            if reasons:
                results.append({
                    "function_id": entry.function_id,
                    "reason": "; ".join(reasons),
                    "current_confidence": entry.confidence,
                    "current_docstring": docstring_text,
                })
                seen.add(entry.function_id)

        return results[: self.max_regenerations]

    @staticmethod
    def build_improved_prompt(
        entry: dict,
        validation_issues: list[dict] | None = None,
    ) -> str:
        """Build an improved regeneration prompt for a weak entry.

        The prompt includes the original function signature (taken from
        ``function_id``) and specific feedback from validation issues so
        the LLM can avoid repeating the same mistakes.

        Parameters
        ----------
        entry:
            A dict as returned by :meth:`find_weak_entries`.
        validation_issues:
            Optional list of issue dicts from
            :class:`~agents.docstring_validator.DocstringValidator`.
            Issues whose ``function_id`` matches *entry* are included as
            inline feedback in the prompt.

        Returns
        -------
        str
            A ready-to-send prompt string.
        """
        fid = entry.get("function_id", "unknown")
        current = entry.get("current_docstring", "")
        reason = entry.get("reason", "")

        # Collect relevant validation feedback
        feedback_lines: list[str] = []
        if validation_issues:
            for issue in validation_issues:
                if issue.get("function_id") == fid:
                    itype = issue.get("issue_type", "")
                    details = issue.get("details", "")
                    if itype == "hallucinated_raises":
                        feedback_lines.append(
                            f"- Do NOT document Raises unless the function contains "
                            f"raise statements. ({details})"
                        )
                    elif itype == "args_mismatch":
                        feedback_lines.append(
                            f"- Fix Args section: {details}"
                        )
                    elif itype == "code_contamination":
                        feedback_lines.append(
                            "- Do NOT include Python code in the docstring. "
                            "Write plain prose only."
                        )
                    elif itype == "stub":
                        feedback_lines.append(
                            "- The previous docstring was too short. "
                            "Write a complete docstring with Args and Returns sections."
                        )

        lines = [
            f"Generate an improved Google-style Python docstring for: {fid}",
            "",
            "Requirements:",
            "- Write plain prose (no code blocks).",
            "- Include a one-sentence summary.",
            "- Include Args: and Returns: sections if applicable.",
            "- Only document Raises: if the function actually raises exceptions.",
        ]

        if feedback_lines:
            lines.append("")
            lines.append("Feedback from previous generation:")
            lines.extend(feedback_lines)

        if reason:
            lines.append("")
            lines.append(f"Previous attempt issue: {reason}")

        if current.strip():
            lines.append("")
            lines.append("Previous docstring (for reference):")
            lines.append(current)

        return "\n".join(lines)
