import json
import logging
import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────
# Compliance Audit Logger for Rapha Enterprise Node
#
# Every training request, data access, and result delivery
# is logged to an append-only audit file. This log is
# designed to satisfy hospital IT compliance reviews.
# ──────────────────────────────────────────────────────────

AUDIT_LOG_DIR = Path(os.getenv("RAPHA_AUDIT_DIR", "/var/log/rapha"))
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit.jsonl"

logger = logging.getLogger("rapha.audit")


def _ensure_log_dir():
    """Create audit log directory if it doesn't exist."""
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _compute_entry_hash(entry: dict, prev_hash: str) -> str:
    """Chain each log entry to the previous one for tamper detection.

    This creates a lightweight hash chain — if any entry is modified
    or deleted, the chain breaks and the tampering is detectable.
    """
    raw = json.dumps(entry, sort_keys=True) + prev_hash
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_last_hash() -> str:
    """Read the hash of the last audit entry, or return genesis hash."""
    if not AUDIT_LOG_FILE.exists():
        return "0" * 64  # Genesis hash
    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                return last.get("entry_hash", "0" * 64)
    except (json.JSONDecodeError, IOError):
        pass
    return "0" * 64


def log_event(
    event_type: str,
    job_id: str,
    org: str,
    details: Optional[dict] = None,
):
    """Append an audit event to the compliance log.

    Args:
        event_type: One of TRAIN_REQUEST, TRAIN_COMPLETE, TRAIN_FAILED,
                    DATA_ACCESS, PROOF_GENERATED, HEALTH_CHECK
        job_id: The unique job identifier
        org: The organisation making the request (from API key)
        details: Additional structured data (dataset_id, duration, etc.)
    """
    _ensure_log_dir()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "job_id": job_id,
        "org": org,
        "details": details or {},
        "node_id": os.getenv("RAPHA_NODE_ID", "node-unnamed"),
    }

    # Hash chain for tamper detection
    prev_hash = _get_last_hash()
    entry["prev_hash"] = prev_hash
    entry["entry_hash"] = _compute_entry_hash(entry, prev_hash)

    # Append-only write
    try:
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"AUDIT [{event_type}] job={job_id} org={org}")
    except IOError as e:
        logger.error(f"Failed to write audit log: {e}")


def verify_audit_chain() -> tuple[bool, int]:
    """Verify the integrity of the entire audit log hash chain.

    Returns:
        (is_valid, entry_count) — True if chain is unbroken
    """
    if not AUDIT_LOG_FILE.exists():
        return True, 0

    prev_hash = "0" * 64
    count = 0

    with open(AUDIT_LOG_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                stored_hash = entry.pop("entry_hash", "")
                expected_hash = _compute_entry_hash(entry, prev_hash)
                if stored_hash != expected_hash:
                    return False, count
                prev_hash = stored_hash
                count += 1
            except (json.JSONDecodeError, KeyError):
                return False, count

    return True, count


def export_audit_log(format: str = "json") -> str:
    """Export the audit log for hospital IT review.

    Args:
        format: 'json' or 'csv'

    Returns:
        The audit log contents as a string
    """
    if not AUDIT_LOG_FILE.exists():
        return "[]" if format == "json" else ""

    with open(AUDIT_LOG_FILE, "r") as f:
        lines = f.readlines()

    if format == "json":
        entries = [json.loads(line) for line in lines]
        return json.dumps(entries, indent=2)

    # CSV format
    if not lines:
        return ""
    headers = list(json.loads(lines[0]).keys())
    csv_lines = [",".join(headers)]
    for line in lines:
        entry = json.loads(line)
        csv_lines.append(",".join(str(entry.get(h, "")) for h in headers))
    return "\n".join(csv_lines)
