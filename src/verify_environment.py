import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

def verify_mps_support():
    print("=" * 50)
    print("Environment & MPS Verification")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    has_mps = torch.backends.mps.is_available()
    has_mps_built = torch.backends.mps.is_built()
    
    print(f"MPS Apple Silicon Support Built: {has_mps_built}")
    print(f"MPS Apple Silicon Available: {has_mps}")
    
    if has_mps:
        device = torch.device("mps")
        print("Success! Using 'mps' as hardware acceleration device.")
        
        # Quick tensor test
        try:
            x = torch.ones(5, device=device)
            print(f"Tensor successfully created on {device}: {x}")
        except Exception as e:
            print(f"Warning: Failed to create tensor on MPS: {e}")
            device = torch.device("cpu")
            
    else:
        print("Warning: MPS is not available! Falling back to 'cpu'.")
        device = torch.device("cpu")
        
    return device


class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_california_housing(device):
    print("\n" + "=" * 50)
    print("California Housing Data Test")
    print("=" * 50)
    
    print("1. Fetching dataset via scikit-learn (OpenML fallback equivalent)...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = housing.data.values
    y = housing.target.values
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Features: {list(housing.feature_names)}")

    print("\n2. Preprocessing Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    print("\n3. Testing simple PyTorch Model on Device...")
    model = SimpleNet(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 5
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    print("Verification completed successfully!")

if __name__ == "__main__":
    device = verify_mps_support()
    test_california_housing(device)
