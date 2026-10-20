import json
from pathlib import Path


def test_one_shot_report_contract():
    rep = json.loads(Path("reports/test_report.json").read_text())
    for k in ("n_test", "acc", "f1", "auc", "confusion", "error_bins"):
        assert k in rep
    assert rep["n_test"] == 572
    cm = rep["confusion"]
    assert sum(sum(r) for r in cm) == rep["n_test"]
    assert Path("reports/figures/confusion_matrix.png").exists()
    for feat, bins in rep["error_bins"].items():
        assert len(bins) >= 2
        for b in bins:
            assert set(("bin", "n_err", "n", "pct")) <= set(b)
