"""
run_report.py — Re-run only the analysis step (s07) after experiment completes.
Useful for regenerating figures/report without re-training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.mnist_geometry_sanity.src import s07_analysis

if __name__ == "__main__":
    print("[run_report] Re-running analysis step only...")
    s07_analysis.run_analysis()
    print("[run_report] Done.")
