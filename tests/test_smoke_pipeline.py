"""End-to-end smoke test on synthetic data (step3 only, tiny bootstrap). Verifies the estimation engine runs."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step3_runs_on_synthetic(tmp_path):
    data = tmp_path / "syn.csv"
    subprocess.run([sys.executable, str(ROOT / "tests/make_synthetic_data.py"), str(data), "1500"], check=True)
    env = dict(os.environ, RV_RAW=str(data), RV_OUT=str(tmp_path / "out"), RV_BOOT="9")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pipeline/step3_primary_plan.py")], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "out" / "STEP3_REPORT.md").exists()
    assert (tmp_path / "out" / "PRIMARY_ANALYSIS_PLAN.md").exists()
