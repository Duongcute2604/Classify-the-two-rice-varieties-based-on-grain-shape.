import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
import joblib

FEATURES = ["Area", "Perimeter", "MajorAxis", "MinorAxis",
            "Eccentricity", "ConvexArea", "Extent"]
SEED = 42


def encode(df):
    X = df[FEATURES]
    y = (df["Class"] == "Osmancik").astype(int)
    return X, y


def metrics(est, X, y):
    p = est.predict(X)
    s = est.predict_proba(X)[:, 1]
    return {"acc": round(float(accuracy_score(y, p)), 4),
            "f1": round(float(f1_score(y, p)), 4),
            "auc": round(float(roc_auc_score(y, s)), 4)}


def main():
    tr = pd.read_csv("data/processed/train.csv")
    va = pd.read_csv("data/processed/val.csv")
    Xtr, ytr = encode(tr)
    Xva, yva = encode(va)

    out = {}
    # Baseline: lớp phổ biến + cây depth=1
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    out["baseline"] = {"val_" + k: v for k, v in metrics(dummy, Xva, yva).items()}
    d1 = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xtr, ytr)
    out["depth1"] = {"val_" + k: v for k, v in metrics(d1, Xva, yva).items()}

    # Thí nghiệm 1: hiệu năng theo max_depth
    depths = list(range(1, 13)) + [None]
    curve = []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        m = metrics(t, Xva, yva)
        curve.append({"max_depth": d, **{"val_" + k: v for k, v in m.items()}})
    out["depth_curve"] = curve
    xs = [str(c["max_depth"]) for c in curve]
    plt.figure()
    plt.plot(xs, [c["val_f1"] for c in curve], marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("val F1")
    plt.title("F1 trên tập validation theo max_depth")
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/figures/depth_curve.png", dpi=120)
    plt.close()

    # Thí nghiệm 2: cắt tỉa ccp_alpha bằng CV 5-fold trên train
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(Xtr, ytr)
    alphas = sorted(set(float(a) for a in path.ccp_alphas))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scored = []
    for a in alphas:
        s = cross_val_score(DecisionTreeClassifier(random_state=SEED, ccp_alpha=a),
                            Xtr, ytr, cv=cv, scoring="f1").mean()
        scored.append((a, float(s)))
    best_alpha = max(scored, key=lambda t: t[1])[0]
    out["ccp_best"] = best_alpha
    plt.figure()
    plt.plot([a for a, _ in scored], [s for _, s in scored], marker="o")
    plt.xlabel("ccp_alpha")
    plt.ylabel("CV F1 (5-fold)")
    plt.title("Chọn ccp_alpha bằng cross-validation")
    plt.savefig("reports/figures/ccp_alpha.png", dpi=120)
    plt.close()
    tree = DecisionTreeClassifier(random_state=SEED, ccp_alpha=best_alpha).fit(Xtr, ytr)
    out["tree"] = {"val_" + k: v for k, v in metrics(tree, Xva, yva).items()}
    out["tree"]["ccp_alpha"] = best_alpha
    out["tree"]["n_leaves"] = int(tree.get_n_leaves())

    # Thí nghiệm 3: rừng ngẫu nhiên
    t0 = time.time()
    forest = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    forest.fit(Xtr, ytr)
    out["forest"] = {"val_" + k: v for k, v in metrics(forest, Xva, yva).items()}
    out["forest"]["train_seconds"] = round(time.time() - t0, 1)

    winner = "forest" if out["forest"]["val_f1"] >= out["tree"]["val_f1"] else "tree"
    out["winner"] = winner
    full = pd.concat([tr, va], ignore_index=True)
    Xf, yf = encode(full)
    final = (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
             if winner == "forest" else
             DecisionTreeClassifier(random_state=SEED, ccp_alpha=best_alpha))
    final.fit(Xf, yf)
    imp = dict(zip(FEATURES, (100 * final.feature_importances_ / final.feature_importances_.sum()).round(2)))
    out["importance"] = {k: float(v) for k, v in imp.items()}

    Path("models").mkdir(exist_ok=True)
    minmax = {c: [float(full[c].min()), float(full[c].max())] for c in FEATURES}
    joblib.dump({"estimator": final, "features": FEATURES, "minmax": minmax,
                 "importance": out["importance"], "seed": SEED},
                "models/pipeline.pkl")
    Path("reports/metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({"baseline_val_f1": out["baseline"]["val_f1"],
                      "tree_val_f1": out["tree"]["val_f1"],
                      "forest_val_f1": out["forest"]["val_f1"],
                      "ccp_best": best_alpha, "winner": winner,
                      "importance": out["importance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
