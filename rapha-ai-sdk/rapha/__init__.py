"""
Rapha AI SDK — Privacy-preserving AI training over medical data.

The official Python SDK for the Rapha Protocol. Train any model
on sensitive health data without the data ever leaving the hospital.

Quick Start:
    from rapha import RaphaClient

    client = RaphaClient(api_key="rp_live_...")
    datasets = client.list_datasets()
    job = client.train(model="biomedbert-base", dataset="diabetes_vitals_v2", epochs=5)
    job.wait()
    print(job.metrics)
"""

from .client import RaphaClient
from .job import TrainingJob
from .catalog import DatasetCatalog, DatasetInfo
from .registry import list_base_models, detect_model_type

__all__ = [
    "RaphaClient",
    "TrainingJob",
    "DatasetCatalog",
    "DatasetInfo",
    "list_base_models",
    "detect_model_type",
]

__version__ = "0.2.0"
