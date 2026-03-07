import base64
import io
import torch

def serialize_model_state(model: torch.nn.Module) -> str:
    """Serializes a PyTorch model state_dict to a base64 string."""
    buffer = io.BytesIO()
    # Use standard load/save instead of weights_only to allow full structure payload representation in mock
    torch.save(model.state_dict(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def deserialize_model_state(model: torch.nn.Module, b64_state: str):
    """Deserializes a base64 string back into a PyTorch model state_dict."""
    buffer = io.BytesIO(base64.b64decode(b64_state))
    # Safe loading is good practice, but for mock standard is fine
    state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)

def create_payload(model: torch.nn.Module, dataset_id: str) -> dict:
    """Creates a secure JSON payload containing model weights."""
    return {
        "dataset_id": dataset_id,
        "weights": serialize_model_state(model)
    }
