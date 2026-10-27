from app import create_app


def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_valid_request():
    c = client()
    body = {"area": 15231, "perimeter": 525.6, "major_axis": 184.0,
            "minor_axis": 102.0, "eccentricity": 0.83,
            "convex_area": 16100, "extent": 0.68}
    r = c.post("/api/rice-classify", json=body)
    assert r.status_code == 200
    d = r.get_json()
    assert d["variety"] in ("Cammeo", "Osmancik")
    assert 0 <= d["probability"] <= 1 and len(d["decision_path"]) == 3


def test_negative_area_rejected():
    c = client()
    body = {"area": -5, "perimeter": 525.6, "major_axis": 184.0,
            "minor_axis": 102.0, "eccentricity": 0.83,
            "convex_area": 16100, "extent": 0.68}
    r = c.post("/api/rice-classify", json=body)
    assert r.status_code == 400


def test_missing_field_rejected():
    c = client()
    r = c.post("/api/rice-classify", json={"area": 15000})
    assert r.status_code == 400


def test_pages_render():
    c = client()
    for p in ("/", "/classify", "/dashboard"):
        assert c.get(p).status_code == 200


def test_out_of_domain_and_uncertain_flags():
    import app as appmod
    c = client()
    mm = appmod.BUNDLE["minmax"]
    lo = {col: v[0] for col, v in mm.items()}
    body = {"area": lo["Area"] * 0.9, "perimeter": lo["Perimeter"],
            "major_axis": lo["MajorAxis"], "minor_axis": lo["MinorAxis"],
            "eccentricity": lo["Eccentricity"],
            "convex_area": lo["ConvexArea"], "extent": lo["Extent"]}
    r = c.post("/api/rice-classify", json=body)
    assert r.status_code == 200
    assert r.get_json()["warning"] == "out_of_domain"


def test_csv_missing_column():
    import io
    c = client()
    r = c.post("/classify", data={"csv": (io.BytesIO(b"Area\n15000\n"), "t.csv")},
               content_type="multipart/form-data")
    assert r.status_code == 400
    assert "Thiếu cột" in r.get_data(as_text=True)
