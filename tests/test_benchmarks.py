import sys
import time
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, accuracy_score

# Graph
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Boost & AutoML
import xgboost as xgb
from tabpfn import TabPFNClassifier


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def test_xgboost(device):
    print("\n" + "=" * 50)
    print("XGBoost Test (California Housing from OpenML)")
    print("=" * 50)
    
    # Fetch from OpenML directly
    import openml
    dataset = openml.datasets.get_dataset('california')
    X, y, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if XGBoost supports MPS specifically via hist tree method
    # It might fall back to CPU if XGB is not compiled with Mac Metal support, which is fine for functionality test.
    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=50, 
        max_depth=5,
        tree_method='hist'
    )
    
    start = time.time()
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    duration = time.time() - start
    
    print(f"XGBoost MSE: {mse:.4f}")
    print(f"XGBoost Time: {duration:.2f}s")
    print("XGBoost operational!")


def test_knn_and_graph_embeddings(device):
    print("\n" + "=" * 50)
    print("KNN & PyTorch Geometric Test (California Housing from OpenML Subset)")
    print("=" * 50)
    # Use a small subset to keep test fast
    import openml
    dataset = openml.datasets.get_dataset('california')
    X, _, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    X = X.values[:1000] 
    
    print("1. Computing KNN adjacency via Scikit-Learn...")
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 2. Build Edge Index for PyTorch Geometric
    source_nodes = np.repeat(np.arange(X.shape[0]), 5)
    target_nodes = indices.flatten()
    
    edge_array = np.array([source_nodes, target_nodes])
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    x_tensor = torch.tensor(X, dtype=torch.float)
    
    # Send graph to device (MPS or CPU)
    edge_index = edge_index.to(device)
    x_tensor = x_tensor.to(device)
    duration_knn = time.time() - start
    print(f"KNN graph constructed with {edge_index.shape[1]} edges. Time: {duration_knn:.2f}s")
    
    print("2. Testing PyG Graph Embedding Layer on Device...")
    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, out_dim)

        def forward(self, x, edge_index):
            return self.conv1(x, edge_index)
            
    try:
        model = SimpleGCN(in_dim=X.shape[1], out_dim=16).to(device)
        start = time.time()
        # Single forward pass for embedding extraction
        embeddings = model(x_tensor, edge_index)
        duration_gcn = time.time() - start
        
        print(f"PyG Embeddings shape: {embeddings.shape}")
        print(f"PyG Forward pass time: {duration_gcn:.4f}s")
        print("Graph Embeddings operational on target device!")
    except Exception as e:
        print(f"Error during PyG testing: {e}")


def test_tabpfn():
    print("\n" + "=" * 50)
    print("TabPFN v2.x Test (Breast Cancer)")
    print("=" * 50)
    # TabPFN is historically classification, falling back to classification dataset to test functionality
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)
    
    try:
        classifier = TabPFNClassifier(device='auto')
        start = time.time()
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        acc = accuracy_score(y_test, preds)
        duration = time.time() - start
        
        print(f"TabPFN Accuracy: {acc:.4f}")
        print(f"TabPFN Time: {duration:.2f}s")
        print("TabPFN operational!")
    except Exception as e:
        print(f"Error during TabPFN testing: {e}")


if __name__ == "__main__":
    dev = get_device()
    print(f"Hardware target for Torch/PyG: {dev}")
    
    test_xgboost(dev)
    test_knn_and_graph_embeddings(dev)
    test_tabpfn()
    print("\nAll benchmark tests executed.")
