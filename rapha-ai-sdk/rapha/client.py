"""
Rapha Client — The primary SDK entry point for AI researchers.

Provides a clean API for:
  - Browsing available datasets and nodes
  - Submitting training jobs with any model format
  - Funding and settling compute jobs via on-chain escrow
  - Monitoring training progress

Usage:
    from rapha import RaphaClient

    client = RaphaClient(api_key="rp_live_...")

    # Browse datasets
    datasets = client.list_datasets()

    # Train with a HuggingFace model
    job = client.train(
        model="microsoft/BiomedNLP-BiomedBERT-base",
        dataset="diabetes_vitals_v2",
        target_node="tokyo_med_01",
        epochs=5,
        learning_rate=1e-4,
    )
    job.wait()
    print(job.metrics)
    job.download_weights("./trained.pt")

    # Settle payment
    client.settle(job)
"""

import requests
import logging

from .registry import prepare_model_payload, list_base_models, detect_model_type
from .catalog import DatasetCatalog, DatasetInfo
from .job import TrainingJob

logger = logging.getLogger("rapha.client")

# Default API endpoint
DEFAULT_NODE_URL = "https://api.rapha.ltd"


class RaphaClient:
    """Main client for the Rapha Protocol.

    Handles model submission, job lifecycle, dataset discovery,
    and on-chain settlement.

    Args:
        api_key: Rapha API key for authentication (rp_live_... or rp_test_...).
        escrow_contract_address: On-chain escrow contract for USDC settlement.
        node_url: Enterprise node URL (defaults to production API).
    """

    def __init__(
        self,
        api_key: str | None = None,
        escrow_contract_address: str = "",
        node_url: str = DEFAULT_NODE_URL,
    ):
        self.api_key = api_key
        self.escrow_contract_address = escrow_contract_address
        self.node_url = node_url.rstrip("/")
        self._catalog = DatasetCatalog(self.node_url, api_key=api_key)
        self._job_id: str | None = None  # Legacy compat

    # ── Headers ───────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    # ── Dataset Discovery ─────────────────────────────────────

    def list_datasets(self, condition: str | None = None) -> list[DatasetInfo]:
        """List available training datasets across enterprise nodes.

        Args:
            condition: Optional filter by medical condition.

        Returns:
            List of DatasetInfo objects with schema and record counts.
        """
        return self._catalog.list_datasets(condition=condition)

    def describe_dataset(self, dataset_id: str) -> DatasetInfo | None:
        """Get detailed schema for a specific dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            DatasetInfo with full schema, or None if not found.
        """
        return self._catalog.describe(dataset_id)

    # ── Model Discovery ───────────────────────────────────────

    @staticmethod
    def list_models() -> list[dict]:
        """List base models pre-cached on Rapha enterprise nodes.

        Returns:
            List of model dicts with id, name, description, param count.
        """
        return list_base_models()

    # ── Job Lifecycle ─────────────────────────────────────────

    def fund_job(self, amount: float) -> str:
        """Lock USDC in the escrow contract to fund a training job.

        Args:
            amount: Amount of USDC to lock.

        Returns:
            A unique job_id for this funded job.
        """
        logger.info(f"Funding job with {amount} USDC at {self.escrow_contract_address}")
        # TODO: Real Web3 interaction — call escrow.fundJob(jobId, amount)
        self._job_id = f"job_{abs(hash(str(amount) + str(id(self))))}"
        return self._job_id

    def train(
        self,
        model,
        dataset: str = "hospital_vitals_v1",
        target_node: str | None = None,
        epochs: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        job_id: str | None = None,
    ) -> TrainingJob:
        """Submit a training job to a Rapha enterprise node.

        Accepts any of:
          - A HuggingFace model ID string (e.g. "microsoft/BiomedNLP-BiomedBERT-base")
          - A PyTorch nn.Module instance
          - A local ONNX file path (e.g. "./my_model.onnx")

        The model is serialized and sent to the enterprise node. Training
        runs on local hospital data behind the firewall — raw data never
        leaves the node.

        Args:
            model: Model to train (HF ID, nn.Module, or ONNX path).
            dataset: Target dataset identifier.
            target_node: Node identifier (auto-selected if None).
            epochs: Number of training epochs (1-100).
            learning_rate: Learning rate for optimizer.
            batch_size: Training batch size.
            job_id: Override job ID (auto-generated from fund_job if None).

        Returns:
            A TrainingJob handle for monitoring and result retrieval.

        Raises:
            ValueError: If no job has been funded.
            TypeError: If model format is not recognized.
        """
        # Resolve job ID
        active_job_id = job_id or self._job_id
        if not active_job_id:
            # Auto-fund a mock job for convenience
            logger.info("No job funded — auto-creating mock job ID")
            active_job_id = self.fund_job(0)

        # Detect and serialize model
        model_type = detect_model_type(model)
        logger.info(f"Preparing {model_type} model for training...")
        model_payload = prepare_model_payload(model)

        # Build training request
        payload = {
            "job_id": active_job_id,
            "dataset_id": dataset,
            "model_payload": model_payload,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }

        if target_node:
            payload["target_node"] = target_node

        # For backward compat: also send 'weights' at top level for old nodes
        if "weights" in model_payload:
            payload["weights"] = model_payload["weights"]

        logger.info(
            f"Dispatching training job {active_job_id} to {self.node_url}/train "
            f"(dataset={dataset}, epochs={epochs})"
        )

        # Submit to enterprise node
        try:
            response = requests.post(
                f"{self.node_url}/train",
                json=payload,
                headers=self._headers(),
                timeout=300,  # Training can take a while
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Training completed synchronously for job {active_job_id}")

                # Create a completed TrainingJob
                job = TrainingJob(active_job_id, self.node_url, self._headers().get("X-API-Key"))
                job.status = "completed"
                job.metrics = {
                    "training_duration_ms": result.get("training_duration_ms", 0),
                    "epochs_completed": result.get("epochs_completed", epochs),
                    "final_loss": result.get("final_loss"),
                }
                job.zk_proof = result.get("zk_proof", "")
                job._updated_weights_b64 = result.get("updated_weights", "")

                # If it's a PyTorch model, update weights in-place (legacy behavior)
                self._maybe_update_pytorch_model(model, result)

                return job

            elif response.status_code == 202:
                # Async job accepted
                result = response.json()
                logger.info(f"Job {active_job_id} accepted for async processing")
                job = TrainingJob(active_job_id, self.node_url, self._headers().get("X-API-Key"))
                job.status = "running"
                return job

            else:
                raise RuntimeError(
                    f"Training request failed ({response.status_code}): {response.text}"
                )

        except requests.ConnectionError:
            logger.error(f"Cannot reach enterprise node at {self.node_url}")
            raise RuntimeError(
                f"Cannot connect to enterprise node at {self.node_url}. "
                "Is the node running? For local testing: node_url='http://127.0.0.1:8000'"
            )

    def _maybe_update_pytorch_model(self, model, result: dict):
        """Update a PyTorch model's weights in-place if applicable."""
        try:
            import torch
            if isinstance(model, torch.nn.Module) and "updated_weights" in result:
                from .packaging import deserialize_model_state
                deserialize_model_state(model, result["updated_weights"])
                logger.info("Model weights updated in-place.")
        except (ImportError, Exception) as e:
            logger.debug(f"In-place weight update skipped: {e}")

    def settle(self, job_or_proof) -> bool:
        """Submit a ZK proof to settle the escrow contract.

        Pays the enterprise node operator for completed training.

        Args:
            job_or_proof: A TrainingJob instance or a ZK proof string.

        Returns:
            True if settlement succeeded.
        """
        if isinstance(job_or_proof, TrainingJob):
            proof = job_or_proof.zk_proof
            job_id = job_or_proof.job_id
        else:
            proof = str(job_or_proof)
            job_id = self._job_id or "unknown"

        logger.info(f"Settling job {job_id} with proof: {proof}")
        # TODO: Real Web3 interaction — call escrow.settleJob(jobId, zkProof, nodeAddress)
        return True
