#!/usr/bin/env python
"""Run MNIST routing curvature experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.mnist_routing_curvature.src.mnist_pipeline import run_mnist
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--n", type=int, default=10000)
    a = p.parse_args()
    run_mnist(None if a.full else a.n)
