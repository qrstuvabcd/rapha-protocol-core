from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from trainer import run_local_training
from db import (
    init_db, create_job, update_job_status, get_job,
    get_logs, append_log, list_datasets, get_dataset,
)
from auth.api_keys import verify_api_key
from audit.logger import log_event, verify_audit_chain, export_audit_log
import logging
import time
import os
import hashlib
import json

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
    version="3.0.0",
    description="Privacy-preserving AI training behind hospital firewalls",
    docs_url=None if os.getenv("RAPHA_ENV") == "production" else "/docs",
    redoc_url=None,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rapha.node")


def build_compute_receipt_hash(job_id: str, dataset_id: str, metrics: dict, updated_weights: str) -> str:
    """Build a deterministic receipt commitment without exposing local records."""
    material = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "metrics": metrics,
        "updated_weights_sha256": hashlib.sha256(updated_weights.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "compute_receipt_sha256:" + hashlib.sha256(encoded).hexdigest()


# ── Request / Response Models ─────────────────────────────

class ModelPayload(BaseModel):
    """Model data sent from the SDK."""
    format: str = Field(default="pytorch", description="Model format: pytorch, onnx, huggingface")
    weights: str | None = Field(default=None, description="Base64-encoded model weights")
    full_model: str | None = Field(default=None, description="Base64-encoded full model (architecture + weights)")
    model_data: str | None = Field(default=None, description="Base64-encoded ONNX model data")
    model_id: str | None = Field(default=None, description="HuggingFace model identifier")
    architecture: str | None = Field(default=None, description="Model architecture name")
    param_count: int | None = Field(default=None, description="Parameter count")
    resolved_locally: bool = Field(default=False, description="Whether HF model was resolved client-side")


class TrainingPayload(BaseModel):
    dataset_id: str = Field(..., description="Target dataset identifier")
    weights: str | None = Field(default=None, description="Base64-encoded model weights (legacy)")
    job_id: str = Field(..., description="Unique job identifier from escrow contract")
    model_payload: ModelPayload | None = Field(default=None, description="Full model payload from SDK v0.2+")
    model_arch: str = Field(
        default="auto",
        description="Model architecture identifier (auto-detected from payload)",
    )
    epochs: int = Field(default=5, ge=1, le=100, description="Training epochs")
    learning_rate: float = Field(default=0.01, gt=0, le=1.0, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size")
    target_node: str | None = Field(default=None, description="Target node identifier")


class TrainingResult(BaseModel):
    updated_weights: str
    zk_proof: str
    job_id: str
    training_duration_ms: int
    epochs_completed: int
    final_loss: float | None = None
    metrics: dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    node_id: str
    audit_chain_valid: bool
    audit_entries: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    metrics: dict = {}
    zk_proof: str | None = None
    updated_weights: str | None = None
    error: str | None = None
    created_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None


class JobLogsResponse(BaseModel):
    job_id: str
    status: str
    logs: list[str] = []


class DatasetResponse(BaseModel):
    id: str
    node_id: str
    name: str
    description: str
    condition: str
    record_count: int
    schema_: list[dict] = Field(default_factory=list, alias="schema")
    data_types: list[str] = []
    created_at: str = ""


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    node_id: str


# ── Lifecycle Events ──────────────────────────────────────

@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Rapha Enterprise Node starting up")
    logger.info(f"Node ID: {os.getenv('RAPHA_NODE_ID', 'node-unnamed')}")
    logger.info("Database initialized (EHR, Jobs, Catalog)")


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint — no auth required (for load balancer probes)."""
    chain_valid, entry_count = verify_audit_chain()
    return HealthResponse(
        status="Rapha Node Active",
        version="3.0.0",
        node_id=os.getenv("RAPHA_NODE_ID", "node-unnamed"),
        audit_chain_valid=chain_valid,
        audit_entries=entry_count,
    )


# ── Dataset Catalog ───────────────────────────────────────

@app.get("/datasets", response_model=DatasetListResponse)
def get_datasets(
    condition: str | None = Query(None, description="Filter by medical condition"),
):
    """List available training datasets on this node.

    No auth required — dataset metadata is public.
    Raw data never leaves the node.
    """
    datasets = list_datasets(condition=condition)
    node_id = os.getenv("RAPHA_NODE_ID", "node-unnamed")
    return DatasetListResponse(
        datasets=[DatasetResponse(**d) for d in datasets],
        node_id=node_id,
    )


@app.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset_detail(dataset_id: str):
    """Get detailed metadata for a specific dataset."""
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return DatasetResponse(**ds)


# ── Training ──────────────────────────────────────────────

@app.post("/train", response_model=TrainingResult)
def receive_payload_and_train(
    payload: TrainingPayload,
    org: str = Depends(verify_api_key),
):
    """Execute a training job on local hospital data.

    Requires valid API key. All requests are audit-logged.
    Raw data never leaves this endpoint — only model weights are returned.

    Accepts models in multiple formats:
      - PyTorch full model (architecture + weights)
      - PyTorch state dict only (legacy — uses MockNet)
      - HuggingFace model ID (server-side download)
    """
    logger.info(f"[{org}] Training request: job={payload.job_id} dataset={payload.dataset_id}")

    # Determine model format for logging
    model_format = "legacy"
    model_arch = payload.model_arch
    if payload.model_payload:
        model_format = payload.model_payload.format
        model_arch = payload.model_payload.architecture or "auto"

    # Register job in database
    create_job(
        job_id=payload.job_id,
        dataset_id=payload.dataset_id,
        model_format=model_format,
        model_architecture=model_arch,
        epochs=payload.epochs,
        learning_rate=payload.learning_rate,
        batch_size=payload.batch_size,
        org=org,
    )

    # Audit: log the incoming request
    log_event(
        event_type="TRAIN_REQUEST",
        job_id=payload.job_id,
        org=org,
        details={
            "dataset_id": payload.dataset_id,
            "model_format": model_format,
            "model_arch": model_arch,
            "epochs": payload.epochs,
        },
    )

    start_time = time.time()

    try:
        # Prepare model payload dict for the trainer
        model_payload_dict = None
        if payload.model_payload:
            model_payload_dict = payload.model_payload.model_dump()

        result = run_local_training(
            b64_model_weights=payload.weights or "",
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
            job_id=payload.job_id,
            model_payload=model_payload_dict,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract results
        updated_weights = result["updated_weights"]
        metrics = result.get("metrics", {})
        final_loss = metrics.get("final_loss")

        zk_proof = build_compute_receipt_hash(
            payload.job_id,
            payload.dataset_id,
            metrics,
            updated_weights,
        )

        # Update job in database
        update_job_status(
            payload.job_id,
            "completed",
            metrics={**metrics, "training_duration_ms": duration_ms},
            zk_proof=zk_proof,
            updated_weights=updated_weights,
        )

        # Audit: log successful completion
        log_event(
            event_type="TRAIN_COMPLETE",
            job_id=payload.job_id,
            org=org,
            details={
                "duration_ms": duration_ms,
                "epochs_completed": payload.epochs,
                "final_loss": final_loss,
                "proof_hash": zk_proof[:32],
            },
        )

        logger.info(f"[{org}] Training complete: job={payload.job_id} duration={duration_ms}ms loss={final_loss}")

        return TrainingResult(
            updated_weights=updated_weights,
            zk_proof=zk_proof,
            job_id=payload.job_id,
            training_duration_ms=duration_ms,
            epochs_completed=payload.epochs,
            final_loss=final_loss,
            metrics=metrics,
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)

        # Update job status
        update_job_status(payload.job_id, "failed", error=str(e))

        # Audit: log failure
        log_event(
            event_type="TRAIN_FAILED",
            job_id=payload.job_id,
            org=org,
            details={"error": str(e), "duration_ms": duration_ms},
        )

        logger.error(f"[{org}] Training failed: job={payload.job_id} error={e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Job Status & Logs ─────────────────────────────────────

@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    org: str = Depends(verify_api_key),
):
    """Get the current status of a training job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        metrics=job.get("metrics", {}),
        zk_proof=job.get("zk_proof"),
        updated_weights=job.get("updated_weights"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@app.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
def get_job_logs(
    job_id: str,
    offset: int = Query(0, ge=0, description="Number of log lines to skip"),
    org: str = Depends(verify_api_key),
):
    """Stream training logs for a job.

    Use offset to poll for new lines incrementally.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    logs = get_logs(job_id, offset=offset)
    return JobLogsResponse(
        job_id=job_id,
        status=job["status"],
        logs=logs,
    )


# ── Audit ─────────────────────────────────────────────────

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
