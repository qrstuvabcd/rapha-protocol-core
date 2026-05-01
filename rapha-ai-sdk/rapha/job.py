"""
Rapha TrainingJob — Async job handle returned by client.train().

Provides status polling, log streaming, metric retrieval, and
updated weight downloads. Wraps the /jobs/* enterprise node API.
"""

import time
import requests
import logging

logger = logging.getLogger("rapha.job")


class TrainingJob:
    """Handle for an in-progress or completed training job.

    Returned by ``RaphaClient.train()``. Provides blocking and
    non-blocking ways to wait for results, stream logs, and
    retrieve trained weights.

    Attributes:
        job_id: Unique identifier for this training job.
        node_url: Base URL of the enterprise node running the job.
        status: Current job status (queued, running, completed, failed).
        metrics: Training metrics once the job completes.
        zk_proof: Zero-knowledge proof of computation.
    """

    def __init__(self, job_id: str, node_url: str, api_key: str | None = None):
        self.job_id = job_id
        self.node_url = node_url.rstrip("/")
        self._api_key = api_key
        self.status: str = "queued"
        self.metrics: dict = {}
        self.zk_proof: str | None = None
        self._updated_weights_b64: str | None = None
        self._logs: list[str] = []
        self._poll_interval: float = 2.0  # seconds

    # ── Internal helpers ──────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    def _poll_once(self) -> dict:
        """Fetch current job status from the node."""
        resp = requests.get(
            f"{self.node_url}/jobs/{self.job_id}/status",
            headers=self._headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            self.status = data.get("status", self.status)
            self.metrics = data.get("metrics", self.metrics)
            self.zk_proof = data.get("zk_proof", self.zk_proof)
            self._updated_weights_b64 = data.get("updated_weights", self._updated_weights_b64)
            return data
        else:
            logger.warning(f"Status poll failed ({resp.status_code}): {resp.text}")
            return {"status": self.status}

    # ── Public API ────────────────────────────────────────────

    def wait(self, timeout: float = 600, poll_interval: float | None = None) -> "TrainingJob":
        """Block until the job completes or times out.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Override default poll interval (seconds).

        Returns:
            self, for chaining.

        Raises:
            TimeoutError: If the job does not complete within ``timeout``.
            RuntimeError: If the job fails.
        """
        interval = poll_interval or self._poll_interval
        start = time.time()

        while True:
            data = self._poll_once()
            status = data.get("status", "unknown")

            if status == "completed":
                logger.info(f"Job {self.job_id} completed.")
                return self
            elif status == "failed":
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Training job {self.job_id} failed: {error}")

            elapsed = time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Job {self.job_id} did not complete within {timeout}s "
                    f"(last status: {status})"
                )

            time.sleep(interval)

    def stream_logs(self, follow: bool = True):
        """Print training logs to stdout in real time.

        Args:
            follow: If True, keep polling for new log lines until job ends.
                    If False, print only currently available logs.
        """
        seen = 0
        while True:
            try:
                resp = requests.get(
                    f"{self.node_url}/jobs/{self.job_id}/logs",
                    headers=self._headers(),
                    params={"offset": seen},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    lines = data.get("logs", [])
                    for line in lines:
                        print(f"[{self.job_id}] {line}")
                        self._logs.append(line)
                    seen += len(lines)

                    job_status = data.get("status", "running")
                    if job_status in ("completed", "failed"):
                        self.status = job_status
                        break
            except requests.RequestException as e:
                logger.warning(f"Log stream error: {e}")

            if not follow:
                break
            time.sleep(self._poll_interval)

    def refresh(self) -> "TrainingJob":
        """Refresh job status, metrics, and proof from the node."""
        self._poll_once()
        return self

    def download_weights(self, path: str) -> str:
        """Download the trained model weights to a local file.

        Args:
            path: Local file path to save the weights to.

        Returns:
            The path the weights were saved to.

        Raises:
            RuntimeError: If the job has not completed yet.
        """
        if self.status != "completed":
            self.refresh()
        if self.status != "completed":
            raise RuntimeError(
                f"Cannot download weights — job status is '{self.status}'. "
                "Call job.wait() first."
            )

        if self._updated_weights_b64:
            import base64
            weight_bytes = base64.b64decode(self._updated_weights_b64)
            with open(path, "wb") as f:
                f.write(weight_bytes)
            logger.info(f"Weights saved to {path} ({len(weight_bytes)} bytes)")
            return path
        else:
            # Fetch from node endpoint
            resp = requests.get(
                f"{self.node_url}/jobs/{self.job_id}/weights",
                headers=self._headers(),
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to download weights: {resp.text}")

            with open(path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Weights saved to {path} ({len(resp.content)} bytes)")
            return path

    def __repr__(self) -> str:
        return (
            f"TrainingJob(id={self.job_id!r}, status={self.status!r}, "
            f"metrics={self.metrics})"
        )
