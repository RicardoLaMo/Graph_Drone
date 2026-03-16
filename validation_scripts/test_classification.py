import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig

def test_iris():
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    config = GraphDroneConfig(
        problem_type="classification", 
        n_classes=3,
        router=SetRouterConfig(kind="contextual_transformer_router")
    )
    model = GraphDrone(config)
    
    print("Fitting GraphDrone...")
    model.fit(X_train, y_train)
    
    print("Predicting...")
    # predict() should return [N, C] probabilities for classification
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Iris Accuracy: {acc:.4f}")
    
    if acc > 0.8:
        print("Success: Accuracy is acceptable.")
    else:
        print("Failure: Accuracy is too low.")

if __name__ == "__main__":
    test_iris()
