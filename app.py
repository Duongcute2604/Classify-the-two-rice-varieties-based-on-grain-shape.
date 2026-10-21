from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

FIELDS = [("area", "Area", "Diện tích"), ("perimeter", "Perimeter", "Chu vi"),
          ("major_axis", "MajorAxis", "Trục dài"), ("minor_axis", "MinorAxis", "Trục ngắn"),
          ("eccentricity", "Eccentricity", "Độ lệch tâm"),
          ("convex_area", "ConvexArea", "Diện tích lồi"), ("extent", "Extent", "Độ đầy")]
BUNDLE = joblib.load("models/pipeline.pkl")


def create_app(model_path="models/pipeline.pkl"):
    app = Flask(__name__)

    def field_meta():
        meta = []
        for key, col, vi in FIELDS:
            lo, hi = BUNDLE["minmax"][col]
            meta.append({"key": key, "col": col, "vi": vi,
                         "min": round(lo * 0.5, 2),
                         "max": round(hi * 1.5, 2),
                         "val": round((lo + hi) / 2, 2)})
        return meta

    def validate(body):
        vals, errs = {}, []
        for key, col, vi in FIELDS:
            v = body.get(key)
            if v is None or v == "":
                errs.append(f"Thiếu {col} ({vi})")
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                errs.append(f"{col} ({vi}) phải là số")
                continue
            lo, hi = BUNDLE["minmax"][col]
            if not (lo * 0.5 <= v <= hi * 1.5):
                errs.append(f"{col} ({vi}) ngoài miền [{lo:.1f}, {hi:.1f}]")
            vals[col] = v
        return vals, errs

    def decide(vals):
        X = pd.DataFrame([vals])
        est = BUNDLE["estimator"]
        proba = float(est.predict_proba(X)[0][1])
        variety = "Osmancik" if proba > 0.5 else "Cammeo"
        vi = {col: v for _, col, v in FIELDS}
        top = sorted(BUNDLE.get("importance", {}).items(),
                     key=lambda kv: -kv[1])[:3]
        path = [f"{c} ({vi[c]}) = {vals[c]:.2f}, tầm quan trọng {pct:.1f}%"
                for c, pct in top] or \
               [f"{c} ({vi[c]}) = {vals[c]:.2f}" for c in BUNDLE["features"][:3]]
        warn = None
        if any(v < BUNDLE["minmax"][c][0] or v > BUNDLE["minmax"][c][1]
               for c, v in vals.items()):
            warn = "out_of_domain"
        elif 0.45 <= proba <= 0.55:
            warn = "uncertain"
        return {"variety": variety, "probability": round(proba, 4),
                "decision_path": path, "warning": warn}

    @app.post("/api/rice-classify")
    def api_classify():
        body = request.get_json(force=True, silent=True) or {}
        vals, errs = validate(body)
        if errs:
            return jsonify({"error_vi": "; ".join(errs)}), 400
        return jsonify(decide(vals))

    @app.post("/classify")
    def classify_batch():
        f = request.files.get("csv")
        if f is None:
            return render_template("classify.html", fields=field_meta(),
                                   batch_error="Chưa chọn file CSV"), 400
        try:
            df = pd.read_csv(f)
        except Exception:
            return render_template("classify.html", fields=field_meta(),
                                   batch_error="File CSV không đọc được"), 400
        need = {col for _, col, _ in FIELDS}
        if not need.issubset(df.columns):
            missing = sorted(need - set(df.columns))
            return render_template("classify.html", fields=field_meta(),
                                   batch_error=f"Thiếu cột: {', '.join(missing)}"), 400
        rows = []
        for _, r in df.head(200).iterrows():
            body = {k: r.get(col) for k, col, _ in FIELDS}
            vals, errs = validate(body)
            rows.append({"input": dict(r), "error": "; ".join(errs) if errs
                         else None, "result": None if errs else decide(vals)})
        return render_template("classify.html", fields=field_meta(), batch_rows=rows)

    @app.get("/")
    def intro():
        return render_template("intro.html")

    @app.get("/classify")
    def classify():
        return render_template("classify.html", fields=field_meta())

    @app.get("/dashboard")
    def dashboard():
        import json
        from pathlib import Path
        rep = json.loads(Path("reports/test_report.json").read_text())
        met = json.loads(Path("reports/metrics.json").read_text())
        return render_template("dashboard.html", rep=rep, met=met)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
