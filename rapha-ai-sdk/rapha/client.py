import requests
import torch
from .packaging import create_payload, deserialize_model_state

class RaphaClient:
    def __init__(self, escrow_contract_address: str, node_url: str = "https://api.rapha.ltd"):
        self.escrow_contract_address = escrow_contract_address
        self.node_url = node_url
        self.job_id = None

    def fund_job(self, amount: float):
        """Mocks locking USDC to fund a job on the smart contract."""
        print(f"[SDK] Funding job with {amount} USDC at {self.escrow_contract_address}")
        # In a real scenario, this would interact with a Web3 provider to lock funds
        self.job_id = "job_mock_" + str(hash(amount))
        return self.job_id

    def train(self, model: torch.nn.Module, target_dataset_id: str):
        """Packages the model and sends it to the Enterprise Node for training."""
        if not self.job_id:
            raise ValueError("Must fund_job() before training.")
        
        print(f"[SDK] Packaging model for dataset {target_dataset_id}...")
        payload = create_payload(model, target_dataset_id)
        payload["job_id"] = self.job_id

        print(f"[SDK] Dispatching payload to {self.node_url}/train...")
        response = requests.post(f"{self.node_url}/train", json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Node returned error: {response.text}")
            
        result = response.json()
        print(f"[SDK] Received updated weights and ZK proof from node.")
        
        # Update our local model with the trained weights
        deserialize_model_state(model, result["updated_weights"])
        return result.get("zk_proof", "mock_proof_true")

    def settle(self, zk_proof: str):
        """Mocks triggering the smart contract settlement with the ZK proof."""
        print(f"[SDK] Submitting proof {zk_proof} to escrow contract for settlement of {self.job_id}")
        return True
