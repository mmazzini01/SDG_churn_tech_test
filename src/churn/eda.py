"""EDA logic: pure functions, DataFrame in and DataFrame out. No plotting, no I/O."""

import hashlib

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2_contingency, false_discovery_control, mannwhitneyu
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

MISSING = "Missing"
OTHER = "Other"


def assign_families(columns: list[str], families: dict[str, list[str]]) -> pd.Series:
    """Map each column to the family with the longest matching name prefix."""
    out = {}
    for col in columns:
        matches = [
            (len(prefix), family)
            for family, prefixes in families.items()
            for prefix in prefixes
            if col.lower().startswith(prefix.lower())
        ]
        if not matches:
            raise ValueError(f"No family matches column {col!r}")
        out[col] = max(matches)[1]
    return pd.Series(out, name="family")


def conformity_checks(df: pd.DataFrame, allow_negative: list[str], rules: list[str]) -> pd.Series:
    """Rows breaking each domain rule: a negative value in a numeric column outside allow_negative, or a
    cross-column rule (a pandas expression that is True on a violation; NaN never counts as one)."""
    numeric = df.select_dtypes("number").columns.difference(allow_negative)
    checks = {f"{c} < 0": df[c] < 0 for c in numeric} | {rule: df.eval(rule) for rule in rules}
    return pd.Series({name: int(mask.sum()) for name, mask in checks.items()}, name="violations")


def profile(df: pd.DataFrame, families: pd.Series, cfg: dict) -> pd.DataFrame:
    """One row per column: quality metrics plus near-constant and long-tail flags."""
    prof = pd.DataFrame([_column_stats(df[c]) for c in df.columns], index=df.columns)
    prof.insert(0, "family", families.reindex(prof.index))
    prof["near_constant"] = prof["pct_top"] >= 100 * cfg["near_constant_share"]
    prof["long_tail"] = prof["skew"] > cfg["long_tail_skew"]
    return prof


def _column_stats(s: pd.Series) -> dict:
    present = s.dropna()
    # Shares of the top value, zeros and negatives are over the present values.
    stats = {
        "kind": "numeric" if is_numeric_dtype(s) else "categorical",
        "dtype": str(s.dtype),
        "pct_missing": 100 * s.isna().mean(),
        "n_unique": present.nunique(),
        "pct_top": 100 * present.value_counts(normalize=True).iloc[0],
    }
    if is_numeric_dtype(s):
        stats |= {
            "pct_zero": 100 * (present == 0).mean(),
            "pct_negative": 100 * (present < 0).mean(),
            "min": present.min(),
            "median": present.median(),
            "max": present.max(),
            "skew": present.skew(),
        }
    return stats


def missing_blocks(df: pd.DataFrame, min_pct: float) -> pd.DataFrame:
    """Group columns whose missing values fall on exactly the same rows; one row per block."""
    miss = df.loc[:, 100 * df.isna().mean() >= min_pct].isna()
    by_mask: dict[str, list[str]] = {}
    for col in miss.columns:
        mask_id = hashlib.sha1(np.packbits(miss[col].to_numpy()).tobytes()).hexdigest()
        by_mask.setdefault(mask_id, []).append(col)
    blocks = list(by_mask.values())
    out = pd.DataFrame({
        "representative": [cols[0] for cols in blocks],
        "n_columns": [len(cols) for cols in blocks],
        "pct_missing": [100 * miss[cols[0]].mean() for cols in blocks],
        "columns": [", ".join(cols) for cols in blocks],
    })
    return out.sort_values("pct_missing", ascending=False, ignore_index=True)


def flag_vs_churn(flags: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per boolean column (missing value, unknown code, rare value...): churn of flagged vs all other rows,
    their odds ratio, and the chi-square p-value with Benjamini-Hochberg adjustment."""
    rows = []
    for col in flags.columns:
        f = flags[col]
        p_flag, p_rest = y[f].mean(), y[~f].mean()
        rows.append({
            "variable": col,
            "n_flagged": int(f.sum()),
            "pct_flagged": 100 * f.mean(),
            "churn_flagged": p_flag,
            "churn_rest": p_rest,
            "odds_ratio": _odds(p_flag) / _odds(p_rest),
            "p_value": chi2_contingency(pd.crosstab(f, y), correction=False).pvalue,
        })
    out = pd.DataFrame(rows)
    out["p_fdr"] = false_discovery_control(out["p_value"], method="bh")
    return out


def rank_vs_target(df: pd.DataFrame, target: str, families: pd.Series, n_bins: int, seed: int) -> pd.DataFrame:
    """Univariate signal of every column: signed AUC (monotonic) and binned mutual information (any shape)."""
    y = df[target]
    rows = [
        {
            "variable": col,
            "family": families[col],
            "kind": "numeric" if is_numeric_dtype(df[col]) else "categorical",
            "auc": _auc(df[col], y, seed),
            # Categoricals keep their n_bins most frequent levels (rest -> Other), so every variable gets
            # about n_bins bins: MI grows with the number of bins by chance alone.
            "mi": mutual_info_score(y, _bin(df[col], n_bins, max_levels=n_bins)),
            "coverage": df[col].notna().mean(),
        }
        for col in df.columns.drop(target)
    ]
    rank = pd.DataFrame(rows)
    rank["auc_strength"] = (rank["auc"] - 0.5).abs()
    # Direction only makes sense for numbers: "+" means higher values go with more churn.
    rank["direction"] = np.where(rank["kind"] == "categorical", "", np.where(rank["auc"] > 0.5, "+", "-"))
    rank["rank_auc"] = rank["auc_strength"].rank(ascending=False).astype(int)
    rank["rank_mi"] = rank["mi"].rank(ascending=False).astype(int)
    return rank.sort_values("mi", ascending=False, ignore_index=True)


def _auc(s: pd.Series, y: pd.Series, seed: int) -> float:
    if is_numeric_dtype(s):
        present = s.notna()
        return roc_auc_score(y[present], s[present])
    # Cross-fitted target encoding: a category is never scored with its own rows' churn,
    # so high-cardinality codes (crclscod, 54 levels) don't inflate their AUC.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    encoded = TargetEncoder(target_type="binary", cv=cv).fit_transform(s.astype(object).to_frame(), y)
    return roc_auc_score(y, encoded.ravel())


def tests_vs_target(df: pd.DataFrame, target: str, min_expected: float) -> pd.DataFrame:
    """Effect size and BH-adjusted p-value per column: Mann-Whitney U with rank-biserial correlation for
    numbers (present values only), chi-square with Cramer's V for categoricals (Missing as a level)."""
    y = df[target].astype(bool)
    rows = [{"variable": c} | _test(df[c], y, min_expected) for c in df.columns.drop(target)]
    out = pd.DataFrame(rows)
    out["p_fdr"] = false_discovery_control(out["p_value"], method="bh")
    return out


def _test(s: pd.Series, y: pd.Series, min_expected: float) -> dict:
    if is_numeric_dtype(s):
        churned, stayed = s[y].dropna(), s[~y].dropna()
        res = mannwhitneyu(churned, stayed)
        # Rank-biserial r = 2 * AUC - 1: > 0 means churners tend to have higher values.
        return {"effect_size": 2 * res.statistic / (len(churned) * len(stayed)) - 1, "p_value": res.pvalue,
                "levels_merged": 0}
    levels = s.astype(object).fillna(MISSING)
    counts = levels.value_counts()
    # Chi-square needs an expected count >= min_expected in every cell: rarer levels are pooled, for the test only.
    rare = counts.index[counts * min(y.mean(), 1 - y.mean()) < min_expected]
    res = chi2_contingency(pd.crosstab(levels.where(~levels.isin(rare), OTHER), y), correction=False)
    # Cramer's V with a binary target: sqrt(chi2 / n).
    return {"effect_size": np.sqrt(res.statistic / len(s)), "p_value": res.pvalue, "levels_merged": len(rare)}


def _bin(s: pd.Series, bins: int | list, max_levels: int | None = None) -> pd.Series:
    """Ordered bin label per row (deciles, fixed bands or levels); NaN becomes its own 'Missing' bin."""
    if not is_numeric_dtype(s) or (isinstance(bins, int) and s.nunique() <= bins):
        if max_levels is not None and s.nunique() > max_levels:
            top = s.value_counts().index[:max_levels]
            s = s.where(s.isin(top) | s.isna(), OTHER)
        cat = s.astype("category")
    elif isinstance(bins, int):
        cat = pd.qcut(s, bins, duplicates="drop")
    else:
        # Fixed bands are labelled by their lower edge: "12" means [12, next edge).
        cat = pd.cut(s, bins, right=False, labels=bins[:-1])
    cat = cat.cat.rename_categories(_label)
    return cat.cat.add_categories(MISSING).fillna(MISSING)


def _label(c) -> str:
    """Bin name: deciles as '(left, right]' rounded to 2 decimals, numbers in their shortest form, levels as they are.
    Only the name changes, never the bin."""
    if isinstance(c, pd.Interval):
        # + 0.0 turns -0.0 (e.g. -0.001 rounded) into 0.0, so no bin reads "-0".
        return f"({round(c.left, 2) + 0.0:g}, {round(c.right, 2) + 0.0:g}]"
    return f"{c:g}" if isinstance(c, (int, float)) else str(c)


def churn_by_grid(df: pd.DataFrame, rows: str, row_bins: list, cols: str, col_bins: list, target: str,
                  min_n: int = 0) -> pd.DataFrame:
    """Customers and churn rate per combination of two banded variables; cells under min_n get a NaN rate.
    Missing values are left out (they are analysed separately)."""
    keys = [_bin(df[rows], row_bins).rename(rows), _bin(df[cols], col_bins).rename(cols)]
    grid = df[target].groupby(keys, observed=True).agg(n="size", churn_rate="mean")
    grid = grid.drop(index=MISSING, level=rows, errors="ignore").drop(index=MISSING, level=cols, errors="ignore")
    return grid.assign(churn_rate=grid["churn_rate"].where(grid["n"] >= min_n))


def churn_by_bin(df: pd.DataFrame, col: str, target: str, bins: int | list = 10, min_n: int = 0) -> pd.DataFrame:
    """Churn rate and odds ratio (vs the whole sample) per bin of one column, with a separate Missing bin.

    The sample's churn rate (~50%) may come from sampling on the target, which is undocumented. The odds
    ratio is unchanged by such sampling, while rates and their ratios are not, so it is valid either way;
    with a low real churn rate it is close to the real relative risk.
    Bins with fewer than min_n customers are dropped: their rate is noise.
    """
    max_levels = bins if isinstance(bins, int) else None
    labels = _bin(df[col], bins, max_levels)
    out = df[target].groupby(labels, observed=True).agg(n="size", churn_rate="mean")
    out["odds_ratio"] = _odds(out["churn_rate"]) / _odds(df[target].mean())
    return out[out["n"] >= min_n].rename_axis("bin").reset_index()


def _odds(p):
    return p / (1 - p)


def pick_representatives(ranking: pd.DataFrame, n: int, must_include: list[str]) -> list[str]:
    """must_include first, then the best variable by MI of each family not yet covered."""
    chosen = list(must_include)
    covered = set(ranking.set_index("variable").loc[chosen, "family"])
    for var, family in ranking.sort_values("mi", ascending=False)[["variable", "family"]].itertuples(index=False):
        if len(chosen) >= n:
            break
        if family not in covered:
            chosen.append(var)
            covered.add(family)
    return chosen


def redundant_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Variable pairs whose |correlation| is at or above the threshold."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["var_a", "var_b", "corr"]
    pairs = pairs[pairs["corr"].abs() >= threshold]
    return pairs.sort_values("corr", key=abs, ascending=False, ignore_index=True)


def redundancy_groups(corr: pd.DataFrame, threshold: float, families: pd.Series) -> pd.DataFrame:
    """Groups of variables chained by |corr| >= threshold (connected components).

    Members can be linked only indirectly (a~b, b~c, a and c weakly), so min_abs_corr reports the weakest
    pair inside the group: a group is a set to inspect, not a set to drop.
    """
    _, labels = connected_components((corr.abs() >= threshold).to_numpy(), directed=False)
    groups = pd.Series(labels, index=corr.index)
    rows = []
    for _, members in groups.groupby(groups):
        if len(members) < 2:
            continue
        cols = members.index
        within = corr.loc[cols, cols].abs().to_numpy()[np.triu_indices(len(cols), k=1)]
        rows.append({
            "n_variables": len(cols),
            "min_abs_corr": within.min(),
            "families": ", ".join(sorted(set(families[cols]))),
            "variables": ", ".join(cols),
        })
    return pd.DataFrame(rows).sort_values("n_variables", ascending=False, ignore_index=True)


def removal_candidates(rank: pd.DataFrame, corr: pd.DataFrame, extreme: float, negligible: float) -> pd.DataFrame:
    """Variables the model comparison should try dropping, with the reason; nothing is dropped here.

    Near-duplicate: walking from the highest MI down, a variable with |corr| >= extreme to one already kept,
    so every candidate has a kept stand-in. Negligible signal: |AUC - 0.5| < negligible and MI below the median.
    """
    r = rank.set_index("variable")
    reasons: dict[str, list[str]] = {}
    kept: list[str] = []
    for v in [v for v in r.sort_values("mi", ascending=False).index if v in corr.index]:
        twins = corr.loc[v, kept].abs()
        if (twins >= extreme).any():
            reasons[v] = [f"near-duplicate of {twins.idxmax()} (rho {corr.at[v, twins.idxmax()]:.3f})"]
        else:
            kept.append(v)
    weak = r[(r["auc_strength"] < negligible) & (r["mi"] < r["mi"].median())]
    for v, strength, mi_rank in weak[["auc_strength", "rank_mi"]].itertuples():
        reasons.setdefault(v, []).append(f"negligible signal (|AUC-0.5| {strength:.3f}, MI rank {mi_rank})")
    out = r.loc[list(reasons), ["family", "mi", "auc_strength"]].assign(reason=[" + ".join(x) for x in reasons.values()])
    return out.rename_axis("variable").reset_index().sort_values(["family", "mi"], ascending=[True, False],
                                                                  ignore_index=True)
