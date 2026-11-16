"""Loading, conformity cleaning and hold-out split of the raw dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: Path) -> pd.DataFrame:
    # European number format. Only empty cells are missing: by default pandas also reads the literal "NA"
    # as NaN, but in hnd_webcap it is a level (a handset without web capability), not a missing value.
    return pd.read_csv(path, sep=";", decimal=",", keep_default_na=False, na_values=[""])


def na_tokens(path: Path) -> pd.DataFrame:
    """Literal text of the cells that pandas' default reader turns into NaN: per token, how many cells and
    which columns. It shows what "missing" looks like in the file before deciding what counts as missing."""
    text = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    default = pd.read_csv(path, sep=";", decimal=",")
    cells = text.where(default.isna()).stack().dropna()
    found = pd.DataFrame({"token": cells.map(repr).to_numpy(), "column": cells.index.get_level_values(1)})
    return found.groupby("token").agg(cells=("column", "size"), n_columns=("column", "nunique"),
                                      columns=("column", lambda c: ", ".join(c.unique())))


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fixed row-wise conformity rules: unknown codes become NaN, values below a lower bound are clipped to it.
    No statistics are learned, so it is safe before the split, and running it twice changes nothing."""
    out = df.copy()
    for col, codes in cfg["unknown_codes"].items():
        out[col] = out[col].mask(out[col].isin(codes))
    for col, lower in cfg["clip_lower"].items():
        out[col] = out[col].clip(lower=lower)
    return out


def split(df: pd.DataFrame, target: str, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split. The EDA and the pipeline call it with the same config, so they hold out the
    same customers."""
    return train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
