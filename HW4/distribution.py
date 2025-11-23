#!/usr/bin/env python3
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CAT_UNIQUE_THRESHOLD = 10


def grid_dims(n: int):
    if n <= 0:
        return 1, 1
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols


def placeholder_figure(message: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_binary_pies(df: pd.DataFrame, cols: list[str], out_path: str):
    if len(cols) == 0:
        placeholder_figure("No binary columns found", out_path)
        return

    rows, cols_count = grid_dims(len(cols))
    fig, axes = plt.subplots(rows, cols_count, figsize=(cols_count * 4, rows * 3.2))
    # Normalize axes to 2D array for uniform indexing
    if rows * cols_count == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols_count == 1:
        axes = np.array(axes).reshape(rows, cols_count)

    for idx, col in enumerate(cols):
        r, c = divmod(idx, cols_count)
        ax = axes[r, c]
        counts = df[col].dropna().value_counts()
        labels = counts.index.astype(str).tolist()
        ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title(col, fontsize=10)

    # Hide any unused subplots
    for idx in range(len(cols), rows * cols_count):
        r, c = divmod(idx, cols_count)
        axes[r, c].axis("off")

    fig.suptitle("Binary Columns - Pie Charts", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_category_bars(df: pd.DataFrame, cols: list[str], out_path: str, top_k: int = 10):
    if len(cols) == 0:
        placeholder_figure("No categorical columns found", out_path)
        return

    rows, cols_count = grid_dims(len(cols))
    fig, axes = plt.subplots(rows, cols_count, figsize=(cols_count * 4.5, rows * 3.6))
    if rows * cols_count == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols_count == 1:
        axes = np.array(axes).reshape(rows, cols_count)

    for idx, col in enumerate(cols):
        r, c = divmod(idx, cols_count)
        ax = axes[r, c]
        counts = df[col].dropna().value_counts().head(top_k)
        ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx in range(len(cols), rows * cols_count):
        r, c = divmod(idx, cols_count)
        axes[r, c].axis("off")

    fig.suptitle("Categorical Columns - Bar Charts", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_numeric_hists(df: pd.DataFrame, cols: list[str], out_path: str, bins: int = 30):
    if len(cols) == 0:
        placeholder_figure("No numeric columns found", out_path)
        return

    rows, cols_count = grid_dims(len(cols))
    fig, axes = plt.subplots(rows, cols_count, figsize=(cols_count * 4.5, rows * 3.6))
    if rows * cols_count == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols_count == 1:
        axes = np.array(axes).reshape(rows, cols_count)

    for idx, col in enumerate(cols):
        r, c = divmod(idx, cols_count)
        ax = axes[r, c]
        # Coerce to numeric in case some values are strings; drop NaNs
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(data, bins=bins, color="#F58518", edgecolor="black", linewidth=0.3)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx in range(len(cols), rows * cols_count):
        r, c = divmod(idx, cols_count)
        axes[r, c].axis("off")

    fig.suptitle("Numeric Columns - Histograms", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def detect_column_types(df: pd.DataFrame, cat_unique_threshold: int = 10):
    # Binary: exactly two unique, non-null values (any dtype)
    binary_cols = []
    for col in df.columns:
        uniques = pd.unique(df[col].dropna())
        if len(uniques) == 2:
            binary_cols.append(col)

    # Numeric: pandas numeric dtypes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Numeric (excluding binary)
    numeric_nonbinary = [c for c in numeric_cols if c not in binary_cols]

    # Base categorical: non-numeric, non-binary
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in binary_cols]

    # Additionally treat low-cardinality numeric columns as categorical (excluding binary)
    numeric_low_card = [c for c in numeric_nonbinary if df[c].nunique(dropna=True) <= cat_unique_threshold]
    categorical_cols = sorted(set(categorical_cols + numeric_low_card))

    # Final numeric are numeric nonbinary excluding those treated as categorical
    numeric_final = [c for c in numeric_nonbinary if c not in categorical_cols]

    # Sort for stable ordering
    return sorted(binary_cols), sorted(categorical_cols), sorted(numeric_final)


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "diabetes.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    binary_cols, categorical_cols, numeric_cols = detect_column_types(df, cat_unique_threshold=CAT_UNIQUE_THRESHOLD)

    # Print detected columns
    print(f"Categorical unique threshold: {CAT_UNIQUE_THRESHOLD}")
    print(f"Binary columns ({len(binary_cols)}): {binary_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

    # Output paths
    binary_out = os.path.join(base_dir, "binary.png")
    category_out = os.path.join(base_dir, "category.png")
    numeric_out = os.path.join(base_dir, "numeric.png")

    # Save figures
    save_binary_pies(df, binary_cols, binary_out)
    save_category_bars(df, categorical_cols, category_out)
    save_numeric_hists(df, numeric_cols, numeric_out)

    print("Saved figures:")
    print(f" - {binary_out}")
    print(f" - {category_out}")
    print(f" - {numeric_out}")


if __name__ == "__main__":
    main()
