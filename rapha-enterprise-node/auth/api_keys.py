import hashlib
import hmac
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# ──────────────────────────────────────────────────────────
# API Key Authentication for Rapha Enterprise Node
#
# In production, keys are stored hashed in a database.
# For pilot deployments, keys are loaded from environment.
# ──────────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-Rapha-API-Key", auto_error=False)


def _hash_key(key: str) -> str:
    """Hash an API key for storage comparison (SHA-256)."""
    return hashlib.sha256(key.encode()).hexdigest()


def _load_valid_keys() -> dict[str, str]:
    """Load valid API keys from environment.

    Format: RAPHA_API_KEYS="key1:org_name1,key2:org_name2"
    In production, replace with database lookup.
    """
    raw = os.getenv("RAPHA_API_KEYS", "")
    keys = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            key, org = entry.split(":", 1)
            keys[_hash_key(key.strip())] = org.strip()
    return keys


def generate_api_key(prefix: str = "rk") -> str:
    """Generate a new Rapha API key.

    Format: rk_live_<32 random hex chars>
    """
    return f"{prefix}_live_{secrets.token_hex(16)}"


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """FastAPI dependency that validates the API key.

    Returns the organisation name associated with the key.
    Raises 401 if key is missing, 403 if invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-Rapha-API-Key header.",
        )

    hashed = _hash_key(api_key)
    valid_keys = _load_valid_keys()

    if hashed not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return valid_keys[hashed]
