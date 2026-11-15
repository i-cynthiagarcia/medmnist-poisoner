"""Shared naming helpers for generations and run-specific artifacts."""

import re
from datetime import datetime, timezone


def run_artifact_stem(seed: int, run_id: int | str) -> str:
    """Return a stable, collision-resistant stem for one Flower run."""
    return f"seed-{seed}_run-{run_id}"


def resolve_protocol_id(configured_id: str) -> str:
    """Resolve the configured sweep ID or create one UTC timestamp ID."""
    if configured_id == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"protocol_{timestamp}"
    if not re.fullmatch(r"protocol_\d{8}T\d{6}Z", configured_id):
        raise ValueError(
            "protocol-id must be 'auto' or use the form "
            "'protocol_YYYYMMDDTHHMMSSZ'."
        )
    return configured_id


def validate_generation_id(generation_id: str) -> str:
    """Validate a generation directory name."""
    if not re.fullmatch(r"generation-v\d+", generation_id):
        raise ValueError("generation-id must use the form 'generation-vN'.")
    return generation_id
