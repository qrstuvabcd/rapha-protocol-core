"""
Integration Test — Rapha Protocol End-to-End

Tests the full flow: SDK → Enterprise Node → Training → Settlement
Validates both the legacy API and the new v0.2 model format API.
"""

import sys
import os
import subprocess
import time

# Add the SDK to the python path
sdk_path = os.path.join(os.path.dirname(__file__), 'rapha-ai-sdk')
sys.path.append(sdk_path)

try:
    from rapha.client import RaphaClient
    from rapha.catalog import DatasetCatalog
    from rapha.registry import detect_model_type
    import torch
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def run_integration_test():
    print("=" * 60)
    print("  Rapha Protocol Integration Test Suite v2")
    print("=" * 60)
    
    node_dir = os.path.join(os.path.dirname(__file__), 'rapha-enterprise-node')
    venv_python = os.path.join(sdk_path, "venv", "Scripts", "python.exe")
    
    if not os.path.exists(venv_python):
        print(f"Server python env not found at {venv_python}. Please ensure venv is set up.")
        sys.exit(1)
        
    print("\n[1/6] Starting Enterprise Node Server...")
    server_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=node_dir,
    )
    
    try:
        # Wait for the server to be ready
        time.sleep(5)
        
        # ── Test 1: Model Type Detection ──────────────────
        print("\n[2/6] Testing model type detection...")
        
        class MockNet(torch.nn.Module):
            def __init__(self):
                super(MockNet, self).__init__()
                self.fc1 = torch.nn.Linear(3, 10)
                self.fc2 = torch.nn.Linear(10, 1)
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = MockNet()
        assert detect_model_type(model) == "pytorch", "PyTorch model detection failed"
        assert detect_model_type("bert-base-uncased") == "huggingface", "HF model detection failed"
        print("  ✓ Model type detection: PASS")
        
        # ── Test 2: Dataset Catalog ───────────────────────
        print("\n[3/6] Testing dataset catalog...")
        
        catalog = DatasetCatalog(node_url="http://127.0.0.1:8000")
        datasets = catalog.list_datasets()
        assert len(datasets) > 0, "No datasets returned from catalog"
        print(f"  ✓ Dataset catalog: {len(datasets)} datasets found")
        for ds in datasets:
            print(f"    - {ds.id}: {ds.name} ({ds.record_count} records)")
        
        # ── Test 3: New SDK API — PyTorch Model ───────────
        print("\n[4/6] Testing new SDK API with PyTorch model...")
        
        initial_weights = list(model.parameters())[0].clone().detach()
        
        client = RaphaClient(
            api_key="test_key",
            escrow_contract_address="0xMockAddress",
            node_url="http://127.0.0.1:8000"
        )
        
        # Fund and train
        job_id = client.fund_job(amount=100.0)
        print(f"  Funded job: {job_id}")
        
        job = client.train(
            model=model,
            dataset="hospital_vitals_v1",
            epochs=5,
            learning_rate=0.01,
        )
        
        print(f"  Job status: {job.status}")
        assert job.status == "completed", f"Expected completed, got {job.status}"
        assert job.zk_proof, "No ZK proof received"
        assert job.metrics.get("final_loss") is not None, "No final_loss in metrics"
        print(f"  ✓ Training completed — loss: {job.metrics['final_loss']}")
        
        # Verify weights changed
        updated_weights = list(model.parameters())[0].clone().detach()
        assert not torch.equal(initial_weights, updated_weights), "Weights did not change!"
        print("  ✓ Model weights updated in-place")
        
        # ── Test 4: Settlement ────────────────────────────
        print("\n[5/6] Testing settlement...")
        
        settle_result = client.settle(job)
        assert settle_result, "Settlement failed"
        print(f"  ✓ Settlement with TrainingJob: PASS")
        
        # Also test legacy string-based settlement
        settle_result2 = client.settle(job.zk_proof)
        assert settle_result2, "Legacy settlement failed"
        print(f"  ✓ Settlement with proof string (legacy): PASS")
        
        # ── Test 5: Model Catalog ─────────────────────────
        print("\n[6/6] Testing model catalog...")
        
        models = RaphaClient.list_models()
        assert len(models) > 0, "No base models returned"
        print(f"  ✓ Model catalog: {len(models)} base models available")
        for m in models:
            print(f"    - {m['id']}: {m['name']} ({m['params']} params)")
        
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED ✓")
        print("=" * 60)
            
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nShutting down Enterprise Node Server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    run_integration_test()
