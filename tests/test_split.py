import pandas as pd
from src.split import stratified_split


def test_split_ratio_stratify_noleak():
    df = pd.read_csv("data/processed/rice_clean.csv")
    tr, va, te = stratified_split(df)
    n = len(df)
    assert abs(len(tr) / n - 0.70) < 0.02
    assert abs(len(va) / n - 0.15) < 0.02
    assert abs(len(te) / n - 0.15) < 0.02
    h = lambda d: set(pd.util.hash_pandas_object(d, index=False))
    assert h(tr).isdisjoint(h(va)) and h(tr).isdisjoint(h(te)) and h(va).isdisjoint(h(te))
    base = df["Class"].value_counts(normalize=True)["Cammeo"]
    for d in (tr, va, te):
        p = d["Class"].value_counts(normalize=True)
        assert abs(p["Cammeo"] - base) < 0.03
