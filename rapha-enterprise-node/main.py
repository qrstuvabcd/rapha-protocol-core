from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from trainer import run_local_training
from db import init_db
from auth.api_keys import verify_api_key
from audit.logger import log_event, verify_audit_chain, export_audit_log
import logging
import time
import os

# ──────────────────────────────────────────────────────────
# Rapha Enterprise Node — Production-Hardened API
#
# Deployed behind hospital firewalls. Receives training
# requests from pharma clients via the Rapha SDK, executes
# training on local data, and returns only model weights.
#
# Raw patient data never leaves this node.
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Rapha Enterprise Node",
    version="2.0.0",
    description="Privacy-preserving AI training behind hospital firewalls",
    docs_url=None if os.getenv("RAPHA_ENV") == "production" else "/docs",
    redoc_url=None,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rapha.node")


# ── Request / Response Models ─────────────────────────────

class TrainingPayload(BaseModel):
    dataset_id: str = Field(..., description="Target dataset identifier")
    weights: str = Field(..., description="Base64-encoded model weights")
    job_id: str = Field(..., description="Unique job identifier from escrow contract")
    model_arch: str = Field(
        default="mock_net",
        description="Model architecture identifier (future: ONNX registry)",
    )
    epochs: int = Field(default=5, ge=1, le=100, description="Training epochs")
    learning_rate: float = Field(default=0.01, gt=0, le=1.0, description="Learning rate")


class TrainingResult(BaseModel):
    updated_weights: str
    zk_proof: str
    job_id: str
    training_duration_ms: int
    epochs_completed: int


class HealthResponse(BaseModel):
    status: str
    version: str
    node_id: str
    audit_chain_valid: bool
    audit_entries: int


# ── Lifecycle Events ──────────────────────────────────────

@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Rapha Enterprise Node starting up")
    logger.info(f"Node ID: {os.getenv('RAPHA_NODE_ID', 'node-unnamed')}")
    logger.info("Mock EHR database initialized")


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint — no auth required (for load balancer probes)."""
    chain_valid, entry_count = verify_audit_chain()
    return HealthResponse(
        status="Rapha Node Active",
        version="2.0.0",
        node_id=os.getenv("RAPHA_NODE_ID", "node-unnamed"),
        audit_chain_valid=chain_valid,
        audit_entries=entry_count,
    )


@app.post("/train", response_model=TrainingResult)
def receive_payload_and_train(
    payload: TrainingPayload,
    org: str = Depends(verify_api_key),
):
    """Execute a training job on local hospital data.

    Requires valid API key. All requests are audit-logged.
    Raw data never leaves this endpoint — only model weights are returned.
    """
    logger.info(f"[{org}] Training request: job={payload.job_id} dataset={payload.dataset_id}")

    # Audit: log the incoming request
    log_event(
        event_type="TRAIN_REQUEST",
        job_id=payload.job_id,
        org=org,
        details={
            "dataset_id": payload.dataset_id,
            "model_arch": payload.model_arch,
            "epochs": payload.epochs,
        },
    )

    start_time = time.time()

    try:
        updated_weights = run_local_training(
            payload.weights,
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # TODO: Replace with real ZK proof generation (Risc Zero / SP1)
        zk_proof = f"zk_snark_proof_{payload.job_id}_valid"

        # Audit: log successful completion
        log_event(
            event_type="TRAIN_COMPLETE",
            job_id=payload.job_id,
            org=org,
            details={
                "duration_ms": duration_ms,
                "epochs_completed": payload.epochs,
                "proof_hash": zk_proof[:32],
            },
        )

        logger.info(f"[{org}] Training complete: job={payload.job_id} duration={duration_ms}ms")

        return TrainingResult(
            updated_weights=updated_weights,
            zk_proof=zk_proof,
            job_id=payload.job_id,
            training_duration_ms=duration_ms,
            epochs_completed=payload.epochs,
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)

        # Audit: log failure
        log_event(
            event_type="TRAIN_FAILED",
            job_id=payload.job_id,
            org=org,
            details={"error": str(e), "duration_ms": duration_ms},
        )

        logger.error(f"[{org}] Training failed: job={payload.job_id} error={e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/export")
def get_audit_log(
    format: str = "json",
    org: str = Depends(verify_api_key),
):
    """Export the audit log for compliance review.

    Available formats: json, csv
    """
    log_event(
        event_type="AUDIT_EXPORT",
        job_id="n/a",
        org=org,
        details={"format": format},
    )
    return {"audit_log": export_audit_log(format)}


@app.get("/audit/verify")
def verify_audit(org: str = Depends(verify_api_key)):
    """Verify the integrity of the audit log hash chain.

    Returns whether the chain is unbroken (no entries tampered with or deleted).
    """
    is_valid, count = verify_audit_chain()
    return {
        "chain_valid": is_valid,
        "total_entries": count,
    }
