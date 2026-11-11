"""
Topic label hygiene.

Topic names flow between queues and end up in every sample's metadata, so they
must stay clean at every hop — not just where they are generated.

The failure this exists to prevent: when the seed pool drained, the rotation
fallback queued "Related to: <current topic>". That derived label then became
the parent of the next rotation, compounding into topic names thousands of
characters long ("Related to: Related to: Related to: ...") and making the
TOPIC field useless for every sample produced from it.
"""

from __future__ import annotations

import re

_PREFIX = re.compile(r"^\s*related\s+to\s*:\s*", re.IGNORECASE)
_MAX_TOPIC_LEN = 240
_FALLBACK = "general software engineering"


def leaf_topic(topic: str | None) -> str:
    """Return the underlying topic name with derived prefixes stripped.

    >>> leaf_topic("Related to: Related to: Database internals")
    'Database internals'
    >>> leaf_topic("   ")
    'general software engineering'
    """
    name = (topic or "").strip()
    # Strip repeatedly: a chain can carry dozens of nested prefixes.
    while True:
        stripped = _PREFIX.sub("", name, count=1)
        if stripped == name:
            break
        name = stripped.strip()

    if len(name) > _MAX_TOPIC_LEN:
        name = name[:_MAX_TOPIC_LEN].rstrip()
    return name or _FALLBACK


def is_derived_topic(topic: str | None) -> bool:
    """True when the label carries a rotation prefix instead of a real topic."""
    return bool(_PREFIX.match(topic or ""))
