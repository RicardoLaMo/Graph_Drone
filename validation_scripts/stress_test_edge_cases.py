#!/usr/bin/env python3
"""
Edge Case Stress Test for PC-MoE Multiclass

Tests edge cases:
- Very small datasets (N=50)
- Highly imbalanced classes (95:5)
- Large number of classes (20+)
- High-dimensional data (100+ features)
"""

import numpy as np
import torch
import sys
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.join(os.getcwd(), "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig


def test_small_dataset():
    """Test on very small dataset (N=50)"""
    print("\n" + "="*70)
    print("TEST 1: Very Small Dataset (N=50, 3 classes)")
    print("="*70)

    X, y = make_classification(n_samples=50, n_features=10, n_informative=5,
                               n_classes=3, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    try:
        config = GraphDroneConfig(problem_type="classification", n_classes=3)
        model = GraphDrone(config)
        model.fit(X_train, y_train)
        proba = model.predict(X_test)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, pred)
        print(f"✅ Small dataset test PASSED: Accuracy = {acc:.4f}")
        return True
    except Exception as e:
        print(f"❌ Small dataset test FAILED: {e}")
        return False


def test_imbalanced_classes():
    """Test on highly imbalanced dataset (95:5)"""
    print("\n" + "="*70)
    print("TEST 2: Highly Imbalanced Dataset (95:5 split, 3 classes)")
    print("="*70)

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_classes=3, n_clusters_per_class=1,
                               weights=[0.95, 0.04, 0.01], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Show class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    try:
        config = GraphDroneConfig(problem_type="classification", n_classes=3)
        model = GraphDrone(config)
        model.fit(X_train, y_train)
        proba = model.predict(X_test)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")
        print(f"✅ Imbalanced test PASSED: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
        return True
    except Exception as e:
        print(f"❌ Imbalanced test FAILED: {e}")
        return False


def test_many_classes():
    """Test on dataset with many classes (20 classes)"""
    print("\n" + "="*70)
    print("TEST 3: Many Classes (20 classes, N=2000)")
    print("="*70)

    X, y = make_classification(n_samples=2000, n_features=30, n_informative=15,
                               n_classes=20, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Classes: {len(np.unique(y))}")

    try:
        config = GraphDroneConfig(problem_type="classification", n_classes=20)
        model = GraphDrone(config)
        model.fit(X_train, y_train)
        proba = model.predict(X_test)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, pred)
        print(f"✅ Many classes test PASSED: Accuracy = {acc:.4f}")
        return True
    except Exception as e:
        print(f"❌ Many classes test FAILED: {e}")
        return False


def test_high_dimensional():
    """Test on high-dimensional data (150 features)"""
    print("\n" + "="*70)
    print("TEST 4: High-Dimensional Data (150 features, 5 classes, N=500)")
    print("="*70)

    X, y = make_classification(n_samples=500, n_features=150, n_informative=50,
                               n_classes=5, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train size: {len(X_train)}, Features: {X_train.shape[1]}, Classes: 5")

    try:
        config = GraphDroneConfig(problem_type="classification", n_classes=5)
        model = GraphDrone(config)
        model.fit(X_train, y_train)
        proba = model.predict(X_test)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, pred)
        print(f"✅ High-dimensional test PASSED: Accuracy = {acc:.4f}")
        return True
    except Exception as e:
        print(f"❌ High-dimensional test FAILED: {e}")
        return False


def test_binary_classification():
    """Test binary classification (edge case: 2 classes)"""
    print("\n" + "="*70)
    print("TEST 5: Binary Classification (2 classes, N=500)")
    print("="*70)

    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               n_classes=2, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    try:
        config = GraphDroneConfig(problem_type="classification", n_classes=2)
        model = GraphDrone(config)
        model.fit(X_train, y_train)
        proba = model.predict(X_test)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, pred)

        # Try ROC-AUC if binary
        try:
            roc_auc = roc_auc_score(y_test, proba[:, 1])
            print(f"✅ Binary test PASSED: Accuracy = {acc:.4f}, ROC-AUC = {roc_auc:.4f}")
        except:
            print(f"✅ Binary test PASSED: Accuracy = {acc:.4f}")
        return True
    except Exception as e:
        print(f"❌ Binary test FAILED: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("PC-MoE EDGE CASE STRESS TESTS")
    print("="*70)
    print(f"PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    results = {
        "Small dataset": test_small_dataset(),
        "Imbalanced classes": test_imbalanced_classes(),
        "Many classes": test_many_classes(),
        "High-dimensional": test_high_dimensional(),
        "Binary classification": test_binary_classification(),
    }

    print("\n" + "="*70)
    print("STRESS TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")

    total = len(results)
    passed = sum(results.values())

    print("="*70)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🚀 Implementation is ready for full TabArena benchmark")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("⚠️  Review failures before running full benchmark")
        return 1


if __name__ == "__main__":
    exit(main())
