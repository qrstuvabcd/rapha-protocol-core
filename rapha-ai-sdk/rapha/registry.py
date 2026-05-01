"""
Rapha Model Registry — Handles model format detection, conversion, and serialization.

Supports three input types:
  1. HuggingFace model ID (str like "microsoft/BiomedNLP-BiomedBERT-base")
  2. PyTorch nn.Module instance
  3. Local ONNX file path (str ending in .onnx)

All models are serialized into a transport payload for the enterprise node.
"""

import base64
import io
import os
import logging

logger = logging.getLogger("rapha.registry")

# Supported model format identifiers
FORMAT_PYTORCH = "pytorch"
FORMAT_ONNX = "onnx"
FORMAT_HUGGINGFACE = "huggingface"


def detect_model_type(model) -> str:
    """Detect the type of model input.

    Args:
        model: A string (HuggingFace ID or ONNX path) or a torch.nn.Module.

    Returns:
        One of: 'pytorch', 'onnx', 'huggingface'

    Raises:
        TypeError: If the model type is not recognized.
    """
    if isinstance(model, str):
        if model.endswith(".onnx"):
            if not os.path.exists(model):
                raise FileNotFoundError(f"ONNX model file not found: {model}")
            return FORMAT_ONNX
        else:
            # Treat as HuggingFace model ID (e.g. "microsoft/BiomedNLP-BiomedBERT-base")
            return FORMAT_HUGGINGFACE
    else:
        # Check if it's a PyTorch module
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return FORMAT_PYTORCH
        except ImportError:
            pass

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        "Pass a HuggingFace model ID (str), an ONNX file path (str), "
        "or a PyTorch nn.Module."
    )


def serialize_pytorch_model(model) -> dict:
    """Serialize a PyTorch nn.Module to a transport-ready dict.

    Serializes both the model architecture (via torch.save on the full model)
    and a separate state_dict for lightweight weight transfer.

    Args:
        model: A torch.nn.Module instance.

    Returns:
        dict with 'format', 'weights' (base64), 'architecture' (class name),
        and 'full_model' (base64 of the entire pickled model).
    """
    import torch

    # Serialize state_dict (lightweight)
    state_buffer = io.BytesIO()
    torch.save(model.state_dict(), state_buffer)
    weights_b64 = base64.b64encode(state_buffer.getvalue()).decode("utf-8")

    # Serialize full model (includes architecture)
    model_buffer = io.BytesIO()
    torch.save(model, model_buffer)
    full_model_b64 = base64.b64encode(model_buffer.getvalue()).decode("utf-8")

    arch_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())

    logger.info(
        f"Serialized PyTorch model: {arch_name} "
        f"({param_count:,} params, {len(weights_b64)} bytes encoded)"
    )

    return {
        "format": FORMAT_PYTORCH,
        "weights": weights_b64,
        "full_model": full_model_b64,
        "architecture": arch_name,
        "param_count": param_count,
    }


def serialize_onnx_model(onnx_path: str) -> dict:
    """Serialize a local ONNX model file to a transport-ready dict.

    Args:
        onnx_path: Path to a .onnx file.

    Returns:
        dict with 'format', 'model_data' (base64), and 'filename'.
    """
    with open(onnx_path, "rb") as f:
        model_bytes = f.read()

    model_b64 = base64.b64encode(model_bytes).decode("utf-8")
    filename = os.path.basename(onnx_path)
    size_mb = len(model_bytes) / (1024 * 1024)

    logger.info(f"Serialized ONNX model: {filename} ({size_mb:.1f} MB)")

    return {
        "format": FORMAT_ONNX,
        "model_data": model_b64,
        "filename": filename,
        "size_bytes": len(model_bytes),
    }


def resolve_huggingface_model(model_id: str) -> dict:
    """Resolve a HuggingFace model ID into a transport-ready payload.

    Attempts to download the model using the ``transformers`` library.
    If ``transformers`` is not installed, returns a reference payload
    and lets the enterprise node handle the download.

    Args:
        model_id: A HuggingFace model identifier (e.g. "bert-base-uncased").

    Returns:
        dict with 'format', 'model_id', and optionally 'weights' if locally resolved.
    """
    try:
        from transformers import AutoModel
        import torch

        logger.info(f"Downloading HuggingFace model: {model_id}...")
        hf_model = AutoModel.from_pretrained(model_id)

        # Serialize as PyTorch state_dict
        state_buffer = io.BytesIO()
        torch.save(hf_model.state_dict(), state_buffer)
        weights_b64 = base64.b64encode(state_buffer.getvalue()).decode("utf-8")

        param_count = sum(p.numel() for p in hf_model.parameters())

        logger.info(
            f"Resolved HuggingFace model: {model_id} "
            f"({param_count:,} params)"
        )

        return {
            "format": FORMAT_HUGGINGFACE,
            "model_id": model_id,
            "weights": weights_b64,
            "architecture": hf_model.config.architectures[0] if hf_model.config.architectures else "unknown",
            "param_count": param_count,
            "resolved_locally": True,
        }

    except ImportError:
        logger.info(
            f"transformers not installed — sending model reference '{model_id}' "
            "to node for server-side resolution."
        )
        return {
            "format": FORMAT_HUGGINGFACE,
            "model_id": model_id,
            "resolved_locally": False,
        }


def prepare_model_payload(model) -> dict:
    """Detect model type and serialize to a transport-ready payload.

    This is the main entry point for model preparation. It handles
    all three supported input types and returns a uniform dict
    suitable for JSON transport to the enterprise node.

    Args:
        model: HuggingFace model ID (str), ONNX path (str), or torch.nn.Module.

    Returns:
        dict containing at minimum 'format' and model data.

    Raises:
        TypeError: If model type is not recognized.
        FileNotFoundError: If ONNX path does not exist.
    """
    model_type = detect_model_type(model)

    if model_type == FORMAT_PYTORCH:
        return serialize_pytorch_model(model)
    elif model_type == FORMAT_ONNX:
        return serialize_onnx_model(model)
    elif model_type == FORMAT_HUGGINGFACE:
        return resolve_huggingface_model(model)
    else:
        raise ValueError(f"Unknown model format: {model_type}")


# ── Base Model Catalog (pre-cached on nodes) ─────────────

BASE_MODELS = [
    {
        "id": "rapha-vitals-v1",
        "name": "Rapha Vitals Predictor",
        "description": "Lightweight feedforward network for vital signs prediction.",
        "params": "131",
        "frameworks": ["pytorch"],
    },
    {
        "id": "biomedbert-base",
        "name": "BiomedBERT Base",
        "description": "Microsoft BiomedNLP model pre-trained on PubMed abstracts.",
        "params": "110M",
        "frameworks": ["pytorch", "huggingface"],
        "hf_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    },
    {
        "id": "pubmedgpt-2.7b",
        "name": "PubMedGPT 2.7B",
        "description": "Stanford CRFM biomedical language model.",
        "params": "2.7B",
        "frameworks": ["pytorch", "huggingface"],
        "hf_id": "stanford-crfm/pubmedgpt",
    },
]


def list_base_models() -> list[dict]:
    """Return the catalog of base models pre-cached on Rapha nodes."""
    return BASE_MODELS
