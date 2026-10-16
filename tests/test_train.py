import json
from pathlib import Path
import joblib


def test_artifacts_exist_and_honest():
    assert Path("models/pipeline.pkl").exists()
    m = json.loads(Path("reports/metrics.json").read_text())
    assert m["tree"]["val_f1"] > m["baseline"]["val_f1"]  # vượt baseline mới nhận
    assert Path("reports/figures/depth_curve.png").exists()
    assert Path("reports/figures/ccp_alpha.png").exists()
    bundle = joblib.load("models/pipeline.pkl")
    assert bundle["seed"] == 42 and len(bundle["features"]) == 7
