import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

FEATURES = ["Area", "Perimeter", "MajorAxis", "MinorAxis",
            "Eccentricity", "ConvexArea", "Extent"]


def main():
    bundle = joblib.load("models/pipeline.pkl")
    est, mm = bundle["estimator"], bundle["minmax"]
    te = pd.read_csv("data/processed/test.csv")
    X, y_true = te[FEATURES], (te["Class"] == "Osmancik").astype(int)
    p = est.predict(X)
    proba = est.predict_proba(X)[:, 1]
    rep = {"n_test": len(te),
           "acc": round(float(accuracy_score(y_true, p)), 4),
           "f1": round(float(f1_score(y_true, p)), 4),
           "auc": round(float(roc_auc_score(y_true, proba)), 4),
           "confusion": confusion_matrix(y_true, p).tolist()}
    ConfusionMatrixDisplay.from_predictions(y_true, p,
                                            display_labels=["Cammeo", "Osmancik"])
    plt.savefig("reports/figures/confusion_matrix.png", dpi=120)
    plt.close()
    te = te.assign(pred=p, err=(p != y_true.values).astype(int))
    imp = sorted(bundle.get("importance", {}).items(),
                 key=lambda kv: -kv[1])[:2]
    err_bins = {}
    for feat, _ in (imp or [("Area", 0), ("MajorAxis", 0)]):
        q = pd.qcut(te[feat], 4, duplicates="drop")
        g = te.groupby(q, observed=True)["err"].agg(["sum", "count"])
        err_bins[feat] = [{"bin": str(i), "n_err": int(r["sum"]),
                           "n": int(r["count"]),
                           "pct": round(100 * r["sum"] / r["count"], 1)}
                          for i, r in g.iterrows()]
    rep["error_bins"] = err_bins
    Path("reports/test_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "error_bins"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
