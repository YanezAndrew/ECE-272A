#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_object_dtype, is_categorical_dtype


def to_numeric_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric so all columns are represented in the correlation.
    - Numeric columns kept as-is (bool cast to int).
    - Object/Categorical columns are factorized to integer codes with NaN preserved.
    """
    out = {}
    for col in df.columns:
        s = df[col]
        if is_bool_dtype(s):
            out[col] = s.astype(int)
        elif is_numeric_dtype(s):
            out[col] = s
        elif is_object_dtype(s) or is_categorical_dtype(s):
            codes = pd.factorize(s, sort=True)[0].astype(float)
            # pd.factorize encodes NaN as -1; convert back to NaN for correlation
            codes[codes == -1] = np.nan
            out[col] = pd.Series(codes, index=s.index, name=col)
        else:
            # Fallback: try coercion, otherwise factorize
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() > 0:
                out[col] = coerced
            else:
                codes = pd.factorize(s.astype(str), sort=True)[0].astype(float)
                out[col] = pd.Series(codes, index=s.index, name=col)
    return pd.DataFrame(out)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: str):
    plt.figure(figsize=(14, 12))
    ax = plt.gca()

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Ticks and labels
    cols = corr.columns.tolist()
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    ax.set_title("Correlation Heatmap (All Columns)", fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def compute_target_correlations(df_num: pd.DataFrame, target: str) -> pd.Series:
    """
    Compute Pearson correlation between each feature and the target column.
    Returns a Series indexed by feature name, sorted by absolute correlation descending.
    """
    if target not in df_num.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    target_series = pd.to_numeric(df_num[target], errors="coerce")
    corrs = {}
    for col in df_num.columns:
        if col == target:
            continue
        s = pd.to_numeric(df_num[col], errors="coerce")
        # Compute correlation if we have at least 2 non-NaN points
        if s.notna().sum() >= 2 and target_series.notna().sum() >= 2:
            corrs[col] = s.corr(target_series, method="pearson")
        else:
            corrs[col] = np.nan

    return pd.Series(corrs).sort_values(key=lambda x: x.abs(), ascending=False)


def plot_top_abs_correlations(corr_series: pd.Series, k: int, out_path: str, target_label: str = "Diabetes_binary"):
    """
    Plot a bar chart of the top-k features by absolute correlation magnitude with the target.
    Bars are colored by sign (positive vs negative). Heights are absolute values.
    """
    top_all = corr_series.dropna()
    order = top_all.abs().sort_values(ascending=False).head(k)
    top = top_all.loc[order.index]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    color = "#4C78A8"
    heights = np.abs(top.values)
    x = np.arange(len(top))
    ax.bar(x, heights, color=color)

    ax.set_ylabel(f"Absolute Pearson correlation with {target_label}")
    ax.set_title(f"Top {min(k, len(top))} features by absolute correlation with {target_label}", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=9)

    # Annotate absolute values
    for i, v in enumerate(heights):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "diabetes.csv")
    out_path = os.path.join(base_dir, "heatmap.png")
    bar_out_path = os.path.join(base_dir, "bar_chart.png")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    ncols = df.shape[1]
    print(f"Loaded {csv_path} with {ncols} columns: {list(df.columns)}")
    if ncols != 22:
        print(f"Warning: expected 22 columns but found {ncols}. Proceeding with all available columns.")

    df_num = to_numeric_full(df)

    # Compute Pearson correlation
    corr = df_num.corr(method="pearson", min_periods=1)

    # If any NaNs remain (e.g., constant columns), fill off-diagonal NaNs with 0, keep diagonal as 1
    corr_filled = corr.copy()
    # Set diagonal to 1 (if not already)
    np.fill_diagonal(corr_filled.values, 1.0)
    # Fill remaining NaNs with 0
    corr_filled = corr_filled.fillna(0.0)

    print(f"Correlation matrix shape: {corr_filled.shape}")
    plot_corr_heatmap(corr_filled, out_path)
    print(f"Saved correlation heatmap to: {out_path}")

    # Compute feature correlations with target and plot top 10 by absolute correlation
    target_col = "Diabetes_binary"
    corrs_to_target = compute_target_correlations(df_num, target_col)
    print("Top 10 features by absolute correlation with Diabetes_binary:")
    print(corrs_to_target.head(10))
    plot_top_abs_correlations(corrs_to_target, k=10, out_path=bar_out_path, target_label=target_col)
    print(f"Saved bar chart to: {bar_out_path}")


if __name__ == "__main__":
    main()
