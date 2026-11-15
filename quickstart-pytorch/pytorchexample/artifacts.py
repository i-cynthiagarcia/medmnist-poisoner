"""Naming helpers for run-specific model and result artifacts."""

from datetime import datetime, timezone
import re


def run_artifact_stem(seed: int, run_id: int | str) -> str:
    """Return a stable, collision-resistant stem for one Flower run."""
    return f"seed-{seed}_run-{run_id}"


def resolve_protocol_id(configured_id: str) -> str:
    """Resolve the configured protocol ID or create one UTC timestamp ID."""
    if configured_id == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"protocol_{timestamp}"
    if not re.fullmatch(r"protocol_[A-Za-z0-9._-]+", configured_id):
        raise ValueError(
            "protocol-id must be 'auto' or start with 'protocol_' and "
            "contain only letters, numbers, dots, underscores, and hyphens."
        )
    return configured_id
