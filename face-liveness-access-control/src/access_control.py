"""
src/access_control.py

The final access-control decision, kept intentionally tiny and
isolated in its own module so it's trivial to audit and test.

RULE (exactly as specified):

    if liveness == "SPOOF":
        access = "DENIED"
    elif liveness == "LIVE":
        if recognized_user:
            access = "GRANTED"
        else:
            access = "DENIED"

Access is NEVER granted merely because face recognition matches — the
liveness check must pass first. This function is the single place
that rule is implemented; nothing else in the codebase should
independently decide GRANTED/DENIED. Any liveness_label other than
"LIVE" (including None/unknown) is treated as a failed check and
denies access (fail-closed).
"""

from dataclasses import dataclass
from typing import Optional

GRANTED = "GRANTED"
DENIED = "DENIED"


@dataclass
class AccessDecision:
    access_result: str               # "GRANTED" or "DENIED"
    liveness_result: str             # "LIVE" or "SPOOF" (normalized)
    liveness_confidence: float
    recognition_result: str          # "MATCH", "UNKNOWN", or "N/A"
    recognition_confidence: float
    user_name: Optional[str]         # the recognized user's name, or None
    reason: str                      # short human-readable explanation


def decide_access(
    liveness_label: Optional[str],
    liveness_confidence: float,
    recognized_name: Optional[str],
    recognition_score: float,
) -> AccessDecision:
    """
    Apply the access-control rule described above.

    liveness_label: "LIVE" or "SPOOF" (anything else, including None,
        is treated as a failed/inconclusive liveness check -> DENIED).
    liveness_confidence: the liveness model's confidence, for logging.
    recognized_name: the matched user's name, or None if unrecognized
        (this must already reflect the "only checked when LIVE" rule
        upstream — see app.py — but this function is defensive on its
        own regardless of what the caller passes).
    recognition_score: the best similarity score found, for logging
        (even when the match was below threshold / Unknown).
    """
    normalized_liveness = liveness_label if liveness_label in ("LIVE", "SPOOF") else "UNKNOWN"

    if normalized_liveness != "LIVE":
        return AccessDecision(
            access_result=DENIED,
            liveness_result=normalized_liveness,
            liveness_confidence=liveness_confidence,
            recognition_result="N/A",
            recognition_confidence=0.0,
            user_name=None,
            reason="Liveness check did not pass (spoof or inconclusive) — "
                   "recognition is not evaluated.",
        )

    if recognized_name:
        return AccessDecision(
            access_result=GRANTED,
            liveness_result="LIVE",
            liveness_confidence=liveness_confidence,
            recognition_result="MATCH",
            recognition_confidence=recognition_score,
            user_name=recognized_name,
            reason=f"Live face recognized as '{recognized_name}'.",
        )

    return AccessDecision(
        access_result=DENIED,
        liveness_result="LIVE",
        liveness_confidence=liveness_confidence,
        recognition_result="UNKNOWN",
        recognition_confidence=recognition_score,
        user_name=None,
        reason="Live face detected but not recognized as a registered user.",
    )
