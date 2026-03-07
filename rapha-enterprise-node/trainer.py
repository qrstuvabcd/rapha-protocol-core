import torch
import torch.nn as nn
import torch.optim as optim
from db import get_training_data
import base64
import io

# We share this architectural assumption between SDK and Node for testing
class MockNet(nn.Module):
    def __init__(self):
        super(MockNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def deserialize_model_state(model: torch.nn.Module, b64_state: str):
    buffer = io.BytesIO(base64.b64decode(b64_state))
    state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)

def serialize_model_state(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def run_local_training(b64_model_weights: str) -> str:
    """Deserializes model, trains on local SQLite data, and returns updated weights."""
    model = MockNet()
    deserialize_model_state(model, b64_model_weights)
    
    raw_data = get_training_data()
    features = torch.tensor([[float(row[0]), float(row[1]), 0.0] for row in raw_data], dtype=torch.float32)
    targets = torch.tensor([[float(row[2])] for row in raw_data], dtype=torch.float32)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    print(f"Local training completed. Final Loss: {loss.item()}")
    return serialize_model_state(model)
