# Rapha AI SDK

The official Python SDK for the **Rapha Protocol**—a decentralized "Compute-to-Data" network for AI model training over sensitive health data.

Instead of bringing data to the model, Rapha brings your model to the data using TEEs (Trusted Execution Environments) and ZK-TLS cryptography.

## Installation

Install the package directly from PyPI:

```bash
pip install rapha-ai
```

## Quickstart

### Option A: Train with a Pre-cached Model

```python
from rapha import RaphaClient

client = RaphaClient(api_key="rp_live_...")

# Browse available datasets
datasets = client.list_datasets()
for ds in datasets:
    print(f"{ds.id}: {ds.name} ({ds.record_count:,} records)")

# Train a pre-cached medical model
job = client.train(
    model="rapha-vitals-v1",
    dataset="hospital_vitals_v1",
    epochs=5,
    learning_rate=0.01,
)
job.wait()
print(job.metrics)

# Settle payment to node operator
client.settle(job)
```

### Option B: Bring Your Own PyTorch Model

```python
import torch
import torch.nn as nn
from rapha import RaphaClient

# Define your model
class MedicalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MedicalNet()

# Train on privacy-preserved hospital data
client = RaphaClient(api_key="rp_live_...")
job = client.train(
    model=model,                        # PyTorch nn.Module
    dataset="hospital_vitals_v1",
    epochs=10,
)

job.wait()
print(f"Final loss: {job.metrics['final_loss']}")

# Model weights are updated in-place automatically
print("Training complete — weights updated.")
client.settle(job)
```

### Option C: Drop a HuggingFace Model

```python
from rapha import RaphaClient

client = RaphaClient(api_key="rp_live_...")

job = client.train(
    model="microsoft/BiomedNLP-BiomedBERT-base",   # HuggingFace model ID
    dataset="diabetes_vitals_v2",
    target_node="tokyo_med_01",
    epochs=5,
    learning_rate=1e-4,
)

job.stream_logs()               # Live training logs
job.download_weights("./trained_biomedbert.pt")
client.settle(job)
```

### Option D: Upload an ONNX Model

```python
from rapha import RaphaClient

client = RaphaClient(api_key="rp_live_...")

job = client.train(
    model="./my_custom_model.onnx",     # Local ONNX file
    dataset="cardiac_ecg_v1",
    epochs=20,
)

job.wait()
job.download_weights("./trained_model.pt")
client.settle(job)
```

## Dataset Discovery

```python
from rapha import RaphaClient

client = RaphaClient(api_key="rp_live_...")

# List all datasets
datasets = client.list_datasets()

# Filter by condition
cardio = client.list_datasets(condition="cardiovascular")

# Inspect schema
schema = client.describe_dataset("hospital_vitals_v1")
print(schema.schema)   # [{"field": "blood_pressure_sys", "type": "integer", ...}, ...]
```

## Model Catalog

```python
from rapha import RaphaClient

models = RaphaClient.list_models()
for m in models:
    print(f"{m['id']}: {m['name']} ({m['params']} params)")
```

## How It Works

1. **You define a model** — PyTorch module, HuggingFace ID, or ONNX file.
2. **SDK packages it** — Serialized and encrypted for transport.
3. **Enterprise node trains** — Your model trains on local hospital data behind the firewall. Raw data **never leaves**.
4. **You get back weights + proof** — Updated model weights and a zero-knowledge proof of computation.
5. **Smart contract settles** — The escrow releases USDC to the node operator upon proof verification.

## Local Development

For local testing against the enterprise node:

```python
client = RaphaClient(
    api_key="test_key",
    node_url="http://127.0.0.1:8000"
)
```

## Learn More

Visit [rapha.ltd](https://rapha.ltd) for protocol documentation and enterprise features.
