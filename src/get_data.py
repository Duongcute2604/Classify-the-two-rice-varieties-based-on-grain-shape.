from pathlib import Path
import hashlib
from ucimlrepo import fetch_ucirepo

COLMAP = {"Major_Axis_Length": "MajorAxis", "Minor_Axis_Length": "MinorAxis",
          "Convex_Area": "ConvexArea"}


def main(out="data/raw/rice.csv"):
    ds = fetch_ucirepo(id=545)
    df = ds.data.features.copy()
    df["Class"] = ds.data.targets.iloc[:, 0]
    df = df.rename(columns=COLMAP)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"rows={len(df)} sha={sha}")
    print(df["Class"].value_counts().to_dict())
    return out


if __name__ == "__main__":
    main()
