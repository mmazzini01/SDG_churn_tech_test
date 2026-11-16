import numpy as np
import pandas as pd

from churn import eda
from churn.data import clean, load_raw, split


def test_load_raw_keeps_literal_na_and_reads_decimal_comma(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("hnd_webcap;rev\nNA;1,5\n;2\n", encoding="utf-8")
    df = load_raw(path)
    assert df.loc[0, "hnd_webcap"] == "NA" and pd.isna(df.loc[1, "hnd_webcap"])
    assert df["rev"].tolist() == [1.5, 2.0]


def test_clean_unknown_codes_and_lower_bound_idempotent():
    df = pd.DataFrame({"new_cell": ["U", "Y", "N"], "eqpdays": [-3.0, 5.0, np.nan]})
    cfg = {"unknown_codes": {"new_cell": ["U"]}, "clip_lower": {"eqpdays": 0}}
    out = clean(df, cfg)
    assert out["new_cell"].isna().tolist() == [True, False, False]
    assert out["eqpdays"].tolist()[:2] == [0.0, 5.0] and pd.isna(out.loc[2, "eqpdays"])
    assert df.loc[0, "new_cell"] == "U"  # input left untouched
    pd.testing.assert_frame_equal(clean(out, cfg), out)


def test_split_is_stratified_disjoint_and_reproducible():
    df = pd.DataFrame({"x": range(100), "churn": [1] * 30 + [0] * 70})
    train, test = split(df, "churn", test_size=0.2, seed=42)
    assert (len(train), len(test)) == (80, 20) and not set(train.index) & set(test.index)
    assert train["churn"].mean() == test["churn"].mean() == 0.3
    assert split(df, "churn", test_size=0.2, seed=42)[1].index.equals(test.index)


def test_conformity_checks_negatives_and_rules():
    df = pd.DataFrame({"a": [-1.0, 2.0, np.nan], "diff": [-5.0, 1.0, 0.0], "b": [3.0, 1.0, 1.0]})
    out = eda.conformity_checks(df, allow_negative=["diff"], rules=["a > b"])
    assert out.to_dict() == {"a < 0": 1, "b < 0": 0, "a > b": 1}
