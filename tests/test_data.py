from pathlib import Path
import pandas as pd

EXPECTED = ["Area", "Perimeter", "MajorAxis", "MinorAxis",
            "Eccentricity", "ConvexArea", "Extent", "Class"]


def test_raw_shape_and_columns():
    df = pd.read_csv("data/raw/rice.csv")
    assert list(df.columns) == EXPECTED
    assert len(df) == 3810
    assert set(df["Class"].unique()) == {"Cammeo", "Osmancik"}
    assert df.isna().sum().sum() == 0
