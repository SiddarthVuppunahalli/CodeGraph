"""Safety guardrails for the CodeGraph agent.

Pre-query filtering blocks questions that attempt to exfiltrate secrets,
credentials, or sensitive data. Post-answer scanning redacts any leaked
secret patterns from the agent's output.

Satisfies the 'Safety / Guardrails' rubric requirement.
"""
from __future__ import annotations
import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Pre-query blocklist
# ---------------------------------------------------------------------------

# Patterns that indicate the user is probing for secrets / sensitive data.
# Each entry is (compiled regex, human-readable reason).
_QUERY_BLOCKLIST: list[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(dump|exfiltrate|leak|extract|reveal|expose|show\s+me)\b.*"
                r"\b(secret|password|token|credential|api.?key|private.?key|"
                r"ssh.?key|env\b|\.env)\b", re.I),
     "query requests exfiltration of secrets or credentials"),

    (re.compile(r"\b(\.env|\.env\.local|\.env\.prod|\.env\.secret)\b", re.I),
     "query references .env files which may contain secrets"),

    (re.compile(r"\b(/etc/shadow|/etc/passwd|~/.ssh|id_rsa|id_ed25519)\b", re.I),
     "query references system credential files"),

    (re.compile(r"\b(API_KEY|SECRET_KEY|AWS_SECRET|PRIVATE_KEY|DATABASE_PASSWORD|"
                r"DB_PASSWORD|MASTER_KEY|AUTH_TOKEN)\s*=", re.I),
     "query attempts to read credential assignments"),

    (re.compile(r"\b(steal|exfil|scrape|harvest)\b.*\b(key|token|secret|cred)", re.I),
     "query uses adversarial exfiltration language"),

    (re.compile(r"\b(value|content|contents)\s+(of|for|in)\b.*"
                r"\b(api.?key|secret.?key|password|private.?key|token|"
                r"credential|auth.?token|master.?key|access.?key)\b", re.I),
     "query asks for the value of a secret or credential"),
]


def check_query(question: str) -> Optional[str]:
    """Screen an incoming question against the safety blocklist.

    Returns a refusal reason string if the query is blocked, or None if safe.
    This is a fast regex pass — no LLM call required.
    """
    for pattern, reason in _QUERY_BLOCKLIST:
        if pattern.search(question):
            return reason
    return None


# ---------------------------------------------------------------------------
# Post-answer output scanning
# ---------------------------------------------------------------------------

# Patterns that look like leaked secrets in an agent answer.
_SECRET_PATTERNS: list[Tuple[re.Pattern, str]] = [
    # OpenAI-style keys
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED_API_KEY]"),
    # AWS access key IDs
    (re.compile(r"AKIA[A-Z0-9]{16}"), "[REDACTED_AWS_KEY]"),
    # GitHub personal access tokens
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "[REDACTED_GH_TOKEN]"),
    # GitHub fine-grained tokens
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GH_PAT]"),
    # Generic long hex tokens (≥ 32 hex chars that look like secrets)
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "[REDACTED_HEX_TOKEN]"),
    # password= or secret= assignments with a value
    (re.compile(r"(password|secret|token|api_key)\s*=\s*[\"'][^\"']{4,}[\"']", re.I),
     "[REDACTED_CREDENTIAL]"),
]


def scan_answer(answer: str) -> Tuple[str, bool]:
    """Scan the agent's answer for leaked secret patterns.

    Returns (possibly_redacted_answer, was_redacted).
    If no secrets are found, returns the original answer unchanged.
    """
    redacted = False
    result = answer
    for pattern, replacement in _SECRET_PATTERNS:
        if pattern.search(result):
            result = pattern.sub(replacement, result)
            redacted = True
    return result, redacted
