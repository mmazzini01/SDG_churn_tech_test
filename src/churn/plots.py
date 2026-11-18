"""EDA figures: DataFrame in, matplotlib Figure out. All numbers come from churn.eda."""

import math
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve

from churn.eda import MISSING

# Categorical slots in a fixed order, validated for colour-vision deficiency on adjacent pairs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
STAYED, CHURNED = SERIES[0], SERIES[1]


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": BASELINE, "ytick.color": BASELINE, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": BASELINE, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110,
    })


def family_colors(families: list[str]) -> dict[str, str]:
    """One fixed colour per family, in config order, so a family keeps its colour in every chart."""
    return dict(zip(families, SERIES, strict=True))


def _family_legend(fig: Figure, present: pd.Series, colors: dict[str, str]) -> None:
    handles = [Patch(color=c, label=f) for f, c in colors.items() if f in set(present)]
    fig.legend(handles=handles, loc="upper center", ncols=len(handles), bbox_to_anchor=(0.5, 1.0))


def _average_line(ax: plt.Axes, overall: float, vertical: bool = False) -> None:
    line = ax.axvline if vertical else ax.axhline
    line(100 * overall, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=3)


def plot_missing(prof: pd.DataFrame, colors: dict[str, str], min_pct: float) -> Figure:
    """Share of missing values per column, coloured by family."""
    m = prof[prof["pct_missing"] >= min_pct].sort_values("pct_missing")
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(m) + 1.2))
    ax.barh(m.index, m["pct_missing"], color=m["family"].map(colors), height=0.8, edgecolor=SURFACE)
    for y, v in enumerate(m["pct_missing"]):
        if v >= 10:
            ax.text(v + 0.5, y, f"{v:.0f}%", va="center", color=INK2, fontsize=8)
    ax.grid(axis="x")
    ax.set_xlabel("% of customers with a missing value")
    ax.margins(y=0.01)
    _family_legend(fig, m["family"], colors)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_missing_vs_churn(mvc: pd.DataFrame, overall: float) -> Figure:
    """Dumbbell: churn rate when the value is present vs missing, one row per missing block (flag_vs_churn)."""
    t = mvc.sort_values("churn_flagged").reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.3))
    ax.hlines(y, 100 * t["churn_rest"], 100 * t["churn_flagged"], color=BASELINE, linewidth=2)
    # Same encoding as the decile charts: blue = value present, grey = value missing.
    dots = {"value present": ("churn_rest", SERIES[0]), "value missing": ("churn_flagged", MUTED)}
    for label, (col, color) in dots.items():
        ax.scatter(100 * t[col], y, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=4, label=label)
    _average_line(ax, overall, vertical=True)
    ax.set_yticks(y, [f"{v} ({p:.1f}% missing)" for v, p in zip(t["variable"], t["pct_flagged"])])
    ax.grid(axis="x")
    ax.set_xlabel("churn rate (%) - dashed line: average")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_ranking(rank: pd.DataFrame, colors: dict[str, str], top_n: int) -> Figure:
    """Top variables by mutual information, next to their |AUC - 0.5|, coloured by family."""
    top = rank.head(top_n).iloc[::-1]
    labels = top["variable"] + top["direction"].map({"+": " (+)", "-": " (−)", "": ""})
    bar_colors = top["family"].map(colors)
    fig, (ax_mi, ax_auc) = plt.subplots(1, 2, figsize=(10, 0.24 * top_n + 1.6), sharey=True)
    ax_mi.barh(labels, top["mi"], color=bar_colors, height=0.8, edgecolor=SURFACE)
    ax_auc.barh(labels, top["auc_strength"], color=bar_colors, height=0.8, edgecolor=SURFACE)
    ax_mi.set_title("Mutual information (any shape of effect)")
    ax_auc.set_title("|AUC − 0.5| (monotonic effect)")
    for ax in (ax_mi, ax_auc):
        ax.grid(axis="x")
        ax.margins(y=0.01)
    _family_legend(fig, top["family"], colors)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_churn_by_bin(tables: dict[str, pd.DataFrame], overall: float, ncols: int = 3) -> Figure:
    """Churn rate per bin for several variables, a grey Missing bar, the average as a dashed line
    and the odds ratio of the riskiest bin."""
    nrows = math.ceil(len(tables) / ncols)
    width = 4.4 * ncols if ncols > 1 else 11
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.3 * nrows), sharey=True, squeeze=False)
    for ax, (col, t) in zip(axes.flat, tables.items()):
        x = np.arange(len(t))
        colors = [MUTED if b == MISSING else STAYED for b in t["bin"]]
        ax.bar(x, 100 * t["churn_rate"], color=colors, width=0.8, edgecolor=SURFACE)
        _average_line(ax, overall)
        peak = int(t["odds_ratio"].idxmax())
        ax.annotate(f"OR {t['odds_ratio'][peak]:.2f}", (peak, 100 * t["churn_rate"][peak]), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
        rotate = len(t) > 12 or t["bin"].astype(str).str.len().max() > 6
        ax.set_xticks(x, t["bin"].astype(str), rotation=60 if rotate else 0, ha="right" if rotate else "center",
                      fontsize=7 if rotate else 8)
        ax.set_title(col)
        ax.grid(axis="y")
    for ax in axes.flat[len(tables):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("churn rate (%)")
    fig.tight_layout()
    return fig


def plot_corr(corr: pd.DataFrame) -> Figure:
    """Lower-triangle Spearman heatmap on a blue (positive) / red (negative) diverging scale."""
    cmap = LinearSegmentedColormap.from_list("diverging", [SERIES[7], NEUTRAL, SERIES[0]])
    # Drop the first row and last column: below the diagonal they would be empty.
    lower = corr.iloc[1:, :-1]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.heatmap(lower, mask=np.triu(np.ones(lower.shape, dtype=bool), k=1), cmap=cmap, vmin=-1, vmax=1,
                square=True, linewidths=1, linecolor=SURFACE, cbar_kws={"shrink": 0.6, "label": "Spearman ρ"}, ax=ax)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def plot_churn_grid(grid: pd.DataFrame, overall: float, row_bins: list, col_bins: list) -> Figure:
    """Churn rate per combination of two bands (churn_by_grid), coloured around the average:
    blue = below, orange = above. Each cell shows the rate and the number of customers; blank = too few."""
    rate, n = grid["churn_rate"].unstack(), grid["n"].unstack()
    annot = [[f"{100 * r:.0f}%\nn={k:,.0f}" if pd.notna(r) else "" for r, k in zip(rr, kk)]
             for rr, kk in zip(rate.to_numpy(), n.to_numpy())]
    cmap = LinearSegmentedColormap.from_list("churn", [STAYED, NEUTRAL, CHURNED])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.heatmap(100 * rate, annot=np.array(annot), fmt="", cmap=cmap, center=100 * overall, linewidths=1,
                linecolor=SURFACE, cbar_kws={"label": "churn rate (%)"}, annot_kws={"fontsize": 8}, ax=ax)

    def bands(labels: pd.Index, edges: list) -> list[str]:
        # Bins are labelled by their lower edge; show the full range instead.
        names = {f"{lo:g}": f"{lo}–{hi - 1}" for lo, hi in pairwise(edges)}
        return [names[str(label)] for label in labels]

    ax.set_xticks(np.arange(rate.shape[1]) + 0.5, bands(rate.columns, col_bins), rotation=0)
    ax.set_yticks(np.arange(rate.shape[0]) + 0.5, bands(rate.index, row_bins), rotation=0)
    ax.set_xlabel(f"{rate.columns.name} (band)")
    ax.set_ylabel(f"{rate.index.name} (band)")
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def plot_outliers(df: pd.DataFrame, cols: list[str], target: str) -> Figure:
    """Box plots by churn class on a symlog scale (keeps zeros and negatives visible)."""
    fig, axes = plt.subplots(1, len(cols), figsize=(2.8 * len(cols), 3.8))
    palette = {0: STAYED, 1: CHURNED}
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=target, y=col, hue=target, palette=palette, legend=False, width=0.5, linewidth=1,
                    fliersize=1.5, flierprops={"alpha": 0.3}, ax=ax)
        # Linear only within ±10, so the bulk of the box is not squashed by the band around zero.
        ax.set_yscale("symlog", linthresh=10, linscale=0.5)
        ax.set_xticks([0, 1], ["stayed", "churned"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(col)
        ax.grid(axis="y")
    fig.tight_layout()
    return fig


def plot_cv_path(path: pd.DataFrame, best: pd.Series, chosen: pd.Series, by: str = "l1_ratio") -> Figure:
    """Regularisation path over C, one line per value of `by` (l1_ratio for the elastic net, n_knots for the GAM):
    CV log-loss (± 1 SE) on top, mean number of non-zero coefficients below. Dotted line: best score; solid line:
    the chosen configuration (the same one, unless a rule such as one-SE picks another)."""
    fig, (ax_loss, ax_n) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 2])
    for color, (value, p) in zip(SERIES, path.groupby(by)):
        loss = -p["score"]
        ax_loss.fill_between(p["C"], loss - p["se"], loss + p["se"], color=color, alpha=0.15, linewidth=0)
        ax_loss.plot(p["C"], loss, color=color, linewidth=2, marker="o", markersize=4, label=f"{by} {value:g}")
        ax_n.plot(p["C"], p["n_nonzero"], color=color, linewidth=2, marker="o", markersize=4)
    for ax in (ax_loss, ax_n):
        ax.axvline(best["C"], color=INK2, linewidth=1, linestyle=(0, (1, 2)))
        ax.axvline(chosen["C"], color=INK, linewidth=1)
        ax.grid(axis="y")
    ax_loss.set_ylim(top=-path["score"].min() + 0.002)
    ax_loss.set_ylabel("CV log-loss (lower is better)")
    ax_loss.set_title(f"Chosen: C = {chosen['C']:.4f}, {by} = {chosen[by]:g}, "
                      f"{chosen['n_nonzero']:.0f} non-zero columns (best score: C = {best['C']:.4f})")
    ax_loss.legend(loc="upper right")
    ax_n.set_xscale("log")
    ax_n.set_xlabel("C (inverse regularisation strength, log scale)")
    ax_n.set_ylabel("non-zero columns")
    fig.tight_layout()
    return fig


def plot_importance(imp: pd.DataFrame, colors: dict[str, str] | None, top_n: int) -> Figure:
    """Top variables by mean |log-odds contribution|, coloured by family (one colour when colors is None);
    (+) / (−): higher value, or missing, goes with more / less churn, holding the other variables fixed."""
    top = imp.head(top_n).iloc[::-1]
    labels = top.index + top["direction"].map({"+": " (+)", "-": " (−)", "": ""})
    fig, ax = plt.subplots(figsize=(8, 0.26 * top_n + 1.4))
    color = top["family"].map(colors) if colors else SERIES[0]
    ax.barh(labels, top["importance"], color=color, height=0.8, edgecolor=SURFACE)
    ax.grid(axis="x")
    ax.margins(y=0.01)
    ax.set_xlabel("mean |contribution to the log-odds of churn| across customers")
    if colors:
        _family_legend(fig, top["family"], colors)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if colors else 1))
    return fig


def _percentile(values: pd.DataFrame, var: str, index: pd.Index) -> np.ndarray:
    # Colour by rank, not raw value, so long tails and different units share one scale; NaN where there is no number.
    if var in values and pd.api.types.is_numeric_dtype(values[var]):
        return values.loc[index, var].rank(pct=True).to_numpy()
    return np.full(len(index), np.nan)


def plot_beeswarm(contrib: pd.DataFrame, values: pd.DataFrame, variables: list[str], seed: int) -> Figure:
    """One row per variable (first on top), one dot per customer at its SHAP value, spread vertically by how many
    customers share a similar value. Colour: the customer's value as a percentile within the variable (blue low,
    red high); dark grey: no number (a categorical variable, or missing)."""
    cmap = LinearSegmentedColormap.from_list("value", [SERIES[0], BASELINE, SERIES[7]])
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(variables) + 1.4))
    for row, var in enumerate(reversed(variables)):
        s = contrib[var].to_numpy()
        bins = np.digitize(s, np.linspace(s.min(), s.max(), 60))
        density = pd.Series(bins).map(pd.Series(bins).value_counts()).to_numpy()
        y = row + rng.uniform(-0.4, 0.4, len(s)) * density / density.max()
        pct = _percentile(values, var, contrib.index)
        known = ~np.isnan(pct)
        ax.scatter(s[~known], y[~known], s=2, color=INK2, alpha=0.3, linewidths=0, rasterized=True)
        ax.scatter(s[known], y[known], s=2, c=pct[known], cmap=cmap, vmin=0, vmax=1, alpha=0.5, linewidths=0,
                   rasterized=True)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(range(len(variables)), list(reversed(variables)))
    ax.set_ylim(-0.6, len(variables) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_xlabel("SHAP value: contribution to the log-odds of churn (right = more churn)")
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, shrink=0.5, pad=0.02, ticks=[0, 1])
    bar.ax.set_yticklabels(["low", "high"])
    bar.set_label("value of the variable (percentile)")
    fig.tight_layout()
    return fig


def _binned_mean(x: pd.Series, s: pd.Series) -> pd.DataFrame:
    # Few distinct values (counts, price points): one point per value, so bins never merge two levels.
    bins = x if x.nunique() <= 40 else pd.qcut(x, 40, duplicates="drop")
    return pd.DataFrame({"x": x, "s": s}).groupby(bins, observed=True).mean()


def plot_dependence(contrib: pd.DataFrame, values: pd.DataFrame, variables: list[str], ncols: int = 2,
                    curves: dict[str, pd.DataFrame | list[pd.DataFrame]] | None = None,
                    label: str | None = None) -> Figure:
    """SHAP value against the variable's own value, one panel per variable on a shared y axis: the shape of its effect
    and its size against the others. Black line: mean SHAP per quantile bin. `curves` adds thin lines of other
    contribution tables on the same customers (one colour per key; a list, e.g. fold refits, shares its colour), with
    the black line as `label` in the legend. The x axis is cut at the 1st-99th percentile; missing values are not
    shown."""
    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), sharey=True, squeeze=False)
    for ax, var in zip(axes.flat, variables):
        x, s = values[var], contrib[var]
        keep = x.between(*x.quantile([0.01, 0.99]))
        ax.scatter(x[keep], s[keep], s=3, color=MUTED, alpha=0.25, linewidths=0, rasterized=True)
        for color, (name, tables) in zip(SERIES, (curves or {}).items()):
            for i, other in enumerate(tables if isinstance(tables, list) else [tables]):
                extra = _binned_mean(x[keep], other[var][keep])
                ax.plot(extra["x"], extra["s"], color=color, linewidth=1.2, label=name if i == 0 else None)
        line = _binned_mean(x[keep], s[keep])
        ax.plot(line["x"], line["s"], color=INK, linewidth=2, label=label)
        ax.axhline(0, color=BASELINE, linewidth=1)
        missing = x.isna().mean()
        ax.set_title(var + (f" ({missing:.1%} missing, not shown)" if missing else ""))
        ax.grid(axis="y")
    if curves:
        axes.flat[0].legend(loc="best")
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP value (log-odds)")
    fig.tight_layout()
    return fig


def plot_joint(panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> Figure:
    """One panel per pair (model.joint_effect): the joint log-odds contribution against the x variable, one line per
    value of the second. Solid: the model with the pair's tensor term; dashed, same colour: the additive model, whose
    lines are parallel by construction. Lines that change shape, not just level, are the interaction."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.8 * len(panels)), squeeze=False)
    for ax, (title, (joint, additive)) in zip(axes[:, 0], panels.items()):
        for color, (level, curve) in zip(SERIES, joint.iterrows()):
            ax.plot(joint.columns, curve, color=color, linewidth=2, label=f"{joint.index.name} = {level:g}")
            ax.plot(additive.columns, additive.loc[level], color=color, linewidth=1, linestyle=(0, (4, 3)))
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(f"{title} (dashed: the same lines without the interaction)")
        ax.set_xlabel(joint.columns.name)
        ax.set_ylabel("joint contribution (log-odds)")
        ax.grid(axis="y")
        ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig


def _short(v: object) -> str:
    if pd.isna(v):
        return "missing"
    return f"{v:,.4g}" if isinstance(v, int | float | np.number) else str(v)


def plot_waterfall(contrib: pd.Series, values: pd.Series, base: float, top_n: int = 10) -> Figure:
    """One customer: from the base log-odds (the model's average) to this customer's, one bar per variable, the
    top_n largest first and the rest summed. Orange raises the churn risk, blue lowers it."""
    order = contrib.abs().sort_values(ascending=False).index
    steps = pd.concat([contrib[order[:top_n]],
                       pd.Series({f"other {len(order) - top_n} variables": contrib[order[top_n:]].sum()})])
    start = base + steps.cumsum() - steps
    labels = [f"{v} = {_short(values[v])}" if v in values else v for v in steps.index]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(steps) + 1.6))
    y = np.arange(len(steps))[::-1]
    ax.barh(y, steps, left=start, color=np.where(steps > 0, CHURNED, STAYED), height=0.7, edgecolor=SURFACE)
    for yi, st, d in zip(y, start, steps):
        ax.annotate(f"{d:+.2f}", (st + d if d > 0 else st, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    end = base + steps.sum()
    for x, ls in ((base, (0, (1, 2))), (end, "-")):
        ax.axvline(x, color=INK2, linewidth=1, linestyle=ls)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    # Bars pin the axis to their left edge (sticky edges); unpin it so the base line is not drawn on the frame.
    ax.use_sticky_edges = False
    ax.margins(x=0.06)
    ax.grid(axis="x")
    ax.set_xlabel("log-odds of churn")
    ax.set_title(f"Average customer (dotted): p = {expit(base):.2f}  →  this customer (solid): p = {expit(end):.2f}")
    fig.tight_layout()
    return fig


def plot_pr_curve(y: pd.Series, proba: np.ndarray, threshold: float) -> Figure:
    """Precision against recall over every cut, the chosen cut marked. The dashed line is the churn rate: the
    precision of a random list, so the gap above it is what the model adds."""
    precision, recall, cuts = precision_recall_curve(y, proba)
    at = np.searchsorted(cuts, threshold)  # first cut >= threshold: the point the chosen cut produces
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(recall, precision, color=SERIES[0], linewidth=2, label=f"model (PR-AUC {average_precision_score(y, proba):.3f})")
    ax.axhline(y.mean(), color=INK2, linewidth=1, linestyle=(0, (4, 3)), label=f"random (churn rate {y.mean():.3f})")
    ax.scatter(recall[at], precision[at], s=40, color=CHURNED, zorder=3,
               label=f"cut {threshold:.3f}: precision {precision[at]:.3f}, recall {recall[at]:.3f}")
    ax.set_xlabel("recall: share of churners on the list")
    ax.set_ylabel("precision: share of the list who churn")
    ax.set_xlim(0, 1)
    ax.grid()
    ax.legend(loc="upper right")  # the empty corner: high precision is reached only at low recall
    fig.tight_layout()
    return fig
