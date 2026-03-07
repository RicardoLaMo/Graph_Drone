#!/usr/bin/env python
"""Run California Housing routing curvature experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.california_routing_curvature.src.california_pipeline import run_california

if __name__ == "__main__":
    run_california()
