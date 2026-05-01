"""
Rapha Dataset Catalog — Discovery API for available training datasets.

Researchers use this to browse available datasets on enterprise nodes
before submitting training jobs. Data never leaves the node —
only metadata (schema, record counts, conditions) is returned.
"""

import requests
import logging

logger = logging.getLogger("rapha.catalog")


class DatasetInfo:
    """Metadata about a training dataset hosted on an enterprise node.

    Attributes:
        id: Unique dataset identifier.
        node_id: Enterprise node hosting this dataset.
        node_url: URL of the enterprise node.
        name: Human-readable dataset name.
        description: What this dataset contains.
        condition: Medical condition category.
        record_count: Number of records available.
        schema: List of field descriptors.
        data_types: Types of medical data included.
        created_at: When the dataset was registered.
    """

    def __init__(self, data: dict):
        self.id: str = data.get("id", "")
        self.node_id: str = data.get("node_id", "")
        self.node_url: str = data.get("node_url", "")
        self.name: str = data.get("name", "")
        self.description: str = data.get("description", "")
        self.condition: str = data.get("condition", "")
        self.record_count: int = data.get("record_count", 0)
        self.schema: list[dict] = data.get("schema", [])
        self.data_types: list[str] = data.get("data_types", [])
        self.created_at: str = data.get("created_at", "")

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(id={self.id!r}, name={self.name!r}, "
            f"records={self.record_count:,}, node={self.node_id!r})"
        )


class DatasetCatalog:
    """Client for browsing datasets available across Rapha enterprise nodes.

    Usage:
        catalog = DatasetCatalog(node_url="http://127.0.0.1:8000")
        datasets = catalog.list_datasets()
        schema = catalog.describe("diabetes_vitals_v2")
    """

    def __init__(self, node_url: str, api_key: str | None = None):
        self.node_url = node_url.rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    def list_datasets(self, condition: str | None = None) -> list[DatasetInfo]:
        """List all available datasets, optionally filtered by condition.

        Args:
            condition: Filter by medical condition (e.g. "diabetes").

        Returns:
            List of DatasetInfo objects.
        """
        try:
            params = {}
            if condition:
                params["condition"] = condition

            resp = requests.get(
                f"{self.node_url}/datasets",
                headers=self._headers(),
                params=params,
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                datasets = [DatasetInfo(d) for d in data.get("datasets", [])]
                logger.info(f"Found {len(datasets)} datasets")
                return datasets
            else:
                logger.warning(f"Dataset listing failed ({resp.status_code})")
                return self._fallback_datasets()

        except requests.RequestException as e:
            logger.warning(f"Cannot reach node for dataset catalog: {e}")
            return self._fallback_datasets()

    def describe(self, dataset_id: str) -> DatasetInfo | None:
        """Get detailed metadata about a specific dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            DatasetInfo with full schema, or None if not found.
        """
        try:
            resp = requests.get(
                f"{self.node_url}/datasets/{dataset_id}",
                headers=self._headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                return DatasetInfo(resp.json())
            else:
                logger.warning(f"Dataset {dataset_id} not found ({resp.status_code})")
                return None
        except requests.RequestException as e:
            logger.warning(f"Cannot reach node: {e}")
            return None

    @staticmethod
    def _fallback_datasets() -> list[DatasetInfo]:
        """Return built-in dataset catalog when the node is unreachable.

        This allows SDK users to explore the API even without
        a running enterprise node.
        """
        return [
            DatasetInfo({
                "id": "hospital_vitals_v1",
                "node_id": "tokyo_med_01",
                "name": "Hospital Vitals Dataset v1",
                "description": "Anonymized blood pressure and heart rate readings from 5 patients.",
                "condition": "cardiovascular",
                "record_count": 5,
                "schema": [
                    {"field": "blood_pressure_sys", "type": "integer", "unit": "mmHg"},
                    {"field": "blood_pressure_dia", "type": "integer", "unit": "mmHg"},
                    {"field": "heart_rate", "type": "integer", "unit": "bpm"},
                ],
                "data_types": ["vitals", "blood_pressure"],
                "created_at": "2026-01-15T00:00:00Z",
            }),
            DatasetInfo({
                "id": "diabetes_vitals_v2",
                "node_id": "tokyo_med_01",
                "name": "Diabetes Vitals Study v2",
                "description": "Blood glucose, BMI, and lifestyle metrics for diabetes prediction.",
                "condition": "diabetes",
                "record_count": 12400,
                "schema": [
                    {"field": "blood_glucose", "type": "float", "unit": "mg/dL"},
                    {"field": "bmi", "type": "float", "unit": "kg/m²"},
                    {"field": "hba1c", "type": "float", "unit": "%"},
                    {"field": "age", "type": "integer", "unit": "years"},
                ],
                "data_types": ["vitals", "lab_results"],
                "created_at": "2026-03-01T00:00:00Z",
            }),
            DatasetInfo({
                "id": "cardiac_ecg_v1",
                "node_id": "london_nhs_03",
                "name": "Cardiac ECG Dataset v1",
                "description": "12-lead ECG waveform data for cardiac arrhythmia detection.",
                "condition": "cardiovascular",
                "record_count": 8200,
                "schema": [
                    {"field": "ecg_leads", "type": "array[float]", "unit": "mV"},
                    {"field": "heart_rate", "type": "integer", "unit": "bpm"},
                    {"field": "diagnosis", "type": "string", "unit": "ICD-10"},
                ],
                "data_types": ["ecg", "vitals"],
                "created_at": "2026-02-20T00:00:00Z",
            }),
        ]
