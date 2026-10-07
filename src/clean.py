from pathlib import Path
import pandas as pd

FEATURES = ["Area", "Perimeter", "MajorAxis", "MinorAxis",
            "Eccentricity", "ConvexArea", "Extent"]


def dedupe(df):
    n_dup = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), n_dup


def outlier_report(df):
    rep = {}
    for c in FEATURES:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rep[c] = {"n": n, "pct": round(100 * n / len(df), 2)}
    return rep  # KHÔNG dùng nhãn, KHÔNG loại dòng ở đây


def main():
    df = pd.read_csv("data/raw/rice.csv")
    df, n_dup = dedupe(df)
    print(f"removed_duplicates={n_dup} kept={len(df)}")
    for c, r in outlier_report(df).items():
        print(f"{c}: {r['n']} mẫu ({r['pct']}%) ngoài IQR")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/rice_clean.csv", index=False)


if __name__ == "__main__":
    main()
