"""
Rapha Enterprise Node — Training Pipeline

Handles model deserialization, training on local hospital data,
and result packaging. Supports dynamic model loading for
arbitrary architectures sent by the SDK.

Raw patient data NEVER leaves this module.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from db import get_training_data, append_log, update_job_status
import base64
import io
import hashlib
import logging

logger = logging.getLogger("rapha.trainer")

# ──────────────────────────────────────────────────────────
# Training Pipeline
#
# Supports:
#   1. Full model objects (pickle) — sent by SDK for PyTorch models
#   2. State dict only (legacy) — uses MockNet fallback
#   3. HuggingFace references — future: server-side download
# ──────────────────────────────────────────────────────────


class MockNet(nn.Module):
    """Simple feedforward network for demo training on EHR vitals data."""

    def __init__(self):
        super(MockNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def deserialize_model_state(model: torch.nn.Module, b64_state: str):
    """Deserialize base64-encoded model weights into a PyTorch model."""
    buffer = io.BytesIO(base64.b64decode(b64_state))
    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)


def serialize_model_state(model: torch.nn.Module) -> str:
    """Serialize PyTorch model weights to base64 string."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def compute_weights_hash(b64_weights: str) -> str:
    """Compute SHA-256 hash of model weights for integrity verification."""
    return hashlib.sha256(b64_weights.encode()).hexdigest()


def load_model_from_payload(payload: dict) -> nn.Module:
    """Load a model from the training payload.

    Tries multiple strategies in order:
      1. Full pickled model (from model_payload.full_model)
      2. State dict loaded into MockNet (legacy compat)

    Args:
        payload: The training request payload dict.

    Returns:
        A torch.nn.Module ready for training.
    """
    model_payload = payload.get("model_payload", {})
    model_format = model_payload.get("format", "pytorch")

    # Strategy 1: Full pickled model
    if "full_model" in model_payload:
        try:
            b64_model = model_payload["full_model"]
            buffer = io.BytesIO(base64.b64decode(b64_model))
            model = torch.load(buffer, map_location="cpu", weights_only=False)
            arch = model.__class__.__name__
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Loaded full model: {arch} ({param_count:,} params)")
            return model
        except Exception as e:
            logger.warning(f"Full model deserialization failed: {e}")

    # Strategy 2: HuggingFace model ID — server-side download
    if model_format == "huggingface" and "model_id" in model_payload:
        model_id = model_payload["model_id"]
        try:
            from transformers import AutoModel
            logger.info(f"Downloading HuggingFace model: {model_id}")
            hf_model = AutoModel.from_pretrained(model_id)

            # If weights were sent from client, load them
            if "weights" in model_payload:
                state_buffer = io.BytesIO(base64.b64decode(model_payload["weights"]))
                state_dict = torch.load(state_buffer, map_location="cpu", weights_only=False)
                hf_model.load_state_dict(state_dict)
                logger.info(f"Loaded client-provided weights into {model_id}")

            return hf_model
        except ImportError:
            logger.warning("transformers not installed on node — falling back to MockNet")
        except Exception as e:
            logger.warning(f"HuggingFace model load failed: {e}")

    # Strategy 3: State dict into MockNet (legacy fallback)
    weights_b64 = payload.get("weights") or model_payload.get("weights")
    if weights_b64:
        model = MockNet()
        try:
            deserialize_model_state(model, weights_b64)
            logger.info("Loaded weights into MockNet (legacy path)")
            return model
        except Exception as e:
            logger.warning(f"MockNet weight load failed: {e}")

    # Strategy 4: Fresh MockNet with no pre-loaded weights
    logger.info("Using fresh MockNet (no weights provided)")
    return MockNet()


def run_local_training(
    b64_model_weights: str,
    epochs: int = 5,
    learning_rate: float = 0.01,
    job_id: str | None = None,
    model_payload: dict | None = None,
) -> dict:
    """Train a model on local hospital data and return results.

    The training happens entirely within this node. No raw data leaves.
    Returns a dict with updated weights, metrics, and proof material.

    Args:
        b64_model_weights: Base64-encoded model state dict (legacy).
        epochs: Number of training epochs (1-100).
        learning_rate: SGD learning rate.
        job_id: Optional job ID for log tracking.
        model_payload: Full model payload from SDK (new format).

    Returns:
        dict with 'updated_weights', 'metrics', and metadata.
    """
    # Load the model
    payload = {"weights": b64_model_weights}
    if model_payload:
        payload["model_payload"] = model_payload

    model = load_model_from_payload(payload)

    if job_id:
        update_job_status(job_id, "running")
        append_log(job_id, f"Model loaded: {model.__class__.__name__}")

    # Compute input hash for integrity
    if b64_model_weights:
        input_hash = compute_weights_hash(b64_model_weights)
        logger.info(f"Input weights hash: {input_hash[:16]}...")

    # Load local hospital data (never leaves this process)
    raw_data = get_training_data()
    if not raw_data:
        error_msg = "No training data available in local database"
        if job_id:
            update_job_status(job_id, "failed", error=error_msg)
            append_log(job_id, f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    if job_id:
        append_log(job_id, f"Loaded {len(raw_data)} local records for training")

    features = torch.tensor(
        [[float(row[0]), float(row[1]), 0.0] for row in raw_data],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[float(row[2])] for row in raw_data],
        dtype=torch.float32,
    )

    logger.info(f"Training on {len(raw_data)} local records, {epochs} epochs, lr={learning_rate}")

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # ── Training Loop ─────────────────────────────────────
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        epoch_losses.append(epoch_loss)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            log_msg = f"Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss:.6f}"
            logger.info(f"  {log_msg}")
            if job_id:
                append_log(job_id, log_msg)

    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    logger.info(f"Training complete. Final loss: {final_loss:.6f}")

    # Serialize updated weights
    updated_weights = serialize_model_state(model)
    output_hash = compute_weights_hash(updated_weights)
    logger.info(f"Output weights hash: {output_hash[:16]}...")

    # Build metrics
    metrics = {
        "final_loss": round(final_loss, 6),
        "epoch_losses": [round(l, 6) for l in epoch_losses],
        "records_trained_on": len(raw_data),
        "model_architecture": model.__class__.__name__,
        "param_count": sum(p.numel() for p in model.parameters()),
        "input_hash": input_hash[:16] if b64_model_weights else None,
        "output_hash": output_hash[:16],
    }

    if job_id:
        append_log(job_id, f"Training complete — Final loss: {final_loss:.6f}")

    # For backward compat: return just the weights string if called from legacy path
    return {
        "updated_weights": updated_weights,
        "metrics": metrics,
    }
