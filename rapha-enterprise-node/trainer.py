import torch
import torch.nn as nn
import torch.optim as optim
from db import get_training_data
import base64
import io
import hashlib
import logging

logger = logging.getLogger("rapha.trainer")

# ──────────────────────────────────────────────────────────
# Training Pipeline
#
# Currently supports MockNet for demo/pilot purposes.
# TODO: Add ONNX model registry for arbitrary architectures
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


def run_local_training(
    b64_model_weights: str,
    epochs: int = 5,
    learning_rate: float = 0.01,
) -> str:
    """Deserializes model, trains on local hospital data, returns updated weights.

    The training happens entirely within this node. No raw data leaves.
    Only the updated model weights (base64) are returned to the caller.

    Args:
        b64_model_weights: Base64-encoded model state dict
        epochs: Number of training epochs (1-100)
        learning_rate: SGD learning rate

    Returns:
        Base64-encoded updated model weights
    """
    model = MockNet()
    deserialize_model_state(model, b64_model_weights)

    input_hash = compute_weights_hash(b64_model_weights)
    logger.info(f"Input weights hash: {input_hash[:16]}...")

    # Load local hospital data (never leaves this process)
    raw_data = get_training_data()
    if not raw_data:
        raise ValueError("No training data available in local database")

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

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs} — Loss: {loss.item():.6f}")

    final_loss = loss.item()
    logger.info(f"Training complete. Final loss: {final_loss:.6f}")

    updated_weights = serialize_model_state(model)
    output_hash = compute_weights_hash(updated_weights)
    logger.info(f"Output weights hash: {output_hash[:16]}...")

    return updated_weights
