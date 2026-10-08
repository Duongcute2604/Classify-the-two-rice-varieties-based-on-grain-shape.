import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df, seed=42):
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["Class"],
                               random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["Class"],
                              random_state=seed)
    return (tr.reset_index(drop=True), va.reset_index(drop=True),
            te.reset_index(drop=True))
