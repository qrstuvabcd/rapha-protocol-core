import sys
import os
import subprocess
import time

# Add the SDK to the python path
sdk_path = os.path.join(os.path.dirname(__file__), 'rapha-ai-sdk')
sys.path.append(sdk_path)

try:
    from rapha.client import RaphaClient
    import torch
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def run_integration_test():
    print("Starting Integration Test Loop...")
    
    node_dir = os.path.join(os.path.dirname(__file__), 'rapha-enterprise-node')
    venv_python = os.path.join(sdk_path, "venv", "Scripts", "python.exe")
    
    if not os.path.exists(venv_python):
        print(f"Server python env not found at {venv_python}. Please ensure venv is set up.")
        sys.exit(1)
        
    print("Starting Enterprise Node Server...")
    server_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=node_dir,
    )
    
    try:
        # Wait for the server to be ready
        time.sleep(5)
        
        print("Initializing RaphaClient...")
        class MockNet(torch.nn.Module):
            def __init__(self):
                super(MockNet, self).__init__()
                self.fc1 = torch.nn.Linear(3, 10)
                self.fc2 = torch.nn.Linear(10, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
                
        model = MockNet()
        
        # Capture initial weights
        initial_weights = list(model.parameters())[0].clone().detach()

        client = RaphaClient(escrow_contract_address="0xMockAddress", node_url="http://127.0.0.1:8000")
        
        # Fund the job
        job_id = client.fund_job(amount=100.0)
        print(f"Job funded with ID: {job_id}")
        
        # Trigger Training
        print("Triggering training (SDK -> Enterprise Node)...")
        zk_proof = client.train(model, target_dataset_id="hospital_dataset_1")
        print(f"Training completed. Received ZK Proof: {zk_proof}")
        
        # Verify weights changed
        updated_weights = list(model.parameters())[0].clone().detach()
        if torch.equal(initial_weights, updated_weights):
            raise Exception("Model weights did not change during training!")
            
        print("Model weights were successfully updated locally.")
        
        # Settle Contract
        print("Simulating Smart Contract settlement...")
        settle_result = client.settle(zk_proof)
        if settle_result:
            print("Smart contract successfully settled via ZK proof verification.")
            
        print("\n=== INTEGRATION TEST PASSED! ===")
            
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        
    finally:
        print("Shutting down Enterprise Node Server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    run_integration_test()
