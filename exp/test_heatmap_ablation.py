"""Compatibility test entry point for the refactored ablation package."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablation_heatmap"))
from test_study import HeatmapSweepTests

if __name__ == "__main__":
    unittest.main()
