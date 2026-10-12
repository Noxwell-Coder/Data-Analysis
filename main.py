"""
Customer churn analysis for a retail shop.

Pipeline:
    1. Load and inspect the raw data
    2. Clean it (duplicates, types, invalid values, missing values, outliers)
    3. Analyse which customer segments churn the most
    4. Save charts to figures/ and a model-ready dataset to data/processed/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW_DATA = Path("data/shop_customers.csv")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")

NUMERIC_COLS = [
    "age",
    "months_as_customer",
    "visits_last_3_months",
    "avg_basket_value",
    "online_orders_share_pct",
    "discount_used_pct",
    "days_since_last_visit",
    "late_deliveries",
]
CATEGORICAL_COLS = ["gender", "city", "loyalty_member"]

# Valid ranges for each numeric column; values outside are treated as data-entry errors
VALID_RANGES = {
    "age": (16, 100),
    "months_as_customer": (0, None),
    "visits_last_3_months": (0, None),
    "avg_basket_value": (0, None),
    "online_orders_share_pct": (0, 100),
    "discount_used_pct": (0, 100),
    "days_since_last_visit": (0, None),
    "late_deliveries": (0, None),
}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ----------------------------------------------------------------------------
# 1. Load and inspect
# ----------------------------------------------------------------------------

def load_data(path):
    data = pd.read_csv(path)
    section("1. RAW DATA OVERVIEW")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns\n")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
    return data


# ----------------------------------------------------------------------------
# 2. Clean
# ----------------------------------------------------------------------------

def remove_duplicates(data):
    full_dupes = data.duplicated().sum()
    id_dupes = data["customer_id"].duplicated().sum()
    print(f"Fully duplicated rows: {full_dupes}")
    print(f"Duplicated customer_id values: {id_dupes}")
    return data.drop_duplicates(subset="customer_id", keep="first")


def standardize_categories(data):
    for col in CATEGORICAL_COLS:
        data[col] = data[col].astype(str).str.strip().str.title()
    data["gender"] = data["gender"].str.upper()
    for col in CATEGORICAL_COLS:
        print(f"{col}: {sorted(data[col].unique())}")
    return data


def flag_invalid_values(data):
    """Replace values outside their valid range with NaN so they get imputed."""
    for col, (low, high) in VALID_RANGES.items():
        invalid = pd.Series(False, index=data.index)
        if low is not None:
            invalid |= data[col] < low
        if high is not None:
            invalid |= data[col] > high
        if invalid.any():
            print(f"{col}: {invalid.sum()} invalid values set to NaN")
            data.loc[invalid, col] = pd.NA
    print("Range check complete.")
    return data


def impute_missing(data):
    missing = data.isnull().sum()
    print("Missing values before imputation:")
    print(missing[missing > 0].to_string())

    # Median is robust to skewed distributions and extreme values
    for col in NUMERIC_COLS:
        if data[col].isnull().any():
            median = data[col].median()
            data[col] = data[col].fillna(median)
            print(f"  {col}: filled with median = {median:.2f}")

    assert data.isnull().sum().sum() == 0, "Missing values remain after imputation"
    return data


def report_outliers(data):
    """
    Detect outliers with the IQR rule and report their churn rate.

    Outliers are kept rather than removed: they are plausible values, and in
    days_since_last_visit they are mostly churned customers, so dropping them
    would remove the strongest churn signal from the dataset.
    """
    overall_churn = data["churned"].mean()
    rows = []
    for col in NUMERIC_COLS:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            # e.g. late_deliveries: mostly zeros, so every non-zero value would be flagged
            print(f"{col}: IQR is 0, skipped (discrete / mostly-constant column)")
            continue
        mask = (data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)
        if mask.any():
            rows.append({
                "column": col,
                "outliers": int(mask.sum()),
                "churn_rate_in_outliers": round(data.loc[mask, "churned"].mean(), 2),
            })
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print(f"Overall churn rate: {overall_churn:.2f}")
    print("Decision: outliers are kept (they are valid values and carry churn signal).")
    return report


def clean_data(data):
    section("2. DATA CLEANING")
    print("\n--- Duplicates ---")
    data = remove_duplicates(data)
    print("\n--- Categorical values ---")
    data = standardize_categories(data)
    print("\n--- Invalid values ---")
    data = flag_invalid_values(data)
    print("\n--- Missing values ---")
    data = impute_missing(data)
    print("\n--- Outliers (IQR rule) ---")
    report_outliers(data)

    data["age"] = data["age"].round().astype(int)
    data["churned"] = data["churned"].astype(int)
    print(f"\nClean dataset: {data.shape[0]} rows x {data.shape[1]} columns")
    return data


# ----------------------------------------------------------------------------
# 3. Analyse
# ----------------------------------------------------------------------------

def add_segments(data):
    data["recency_group"] = pd.cut(
        data["days_since_last_visit"],
        bins=[-1, 14, 30, 60, data["days_since_last_visit"].max()],
        labels=["0-14 days", "15-30 days", "31-60 days", "60+ days"],
    )
    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
        right=False,
    )
    return data


def churn_by(data, col):
    return (
        data.groupby(col, observed=True)["churned"]
        .agg(churn_rate="mean", customers="size")
        .round({"churn_rate": 3})
    )


def analyse(data):
    section("3. CHURN ANALYSIS")
    print(f"Overall churn rate: {data['churned'].mean():.1%}\n")

    for col in ["loyalty_member", "recency_group", "late_deliveries", "city", "gender", "age_group"]:
        print(f"--- Churn by {col} ---")
        print(churn_by(data, col).to_string())
        print()

    print("--- Correlation of numeric features with churn ---")
    corr = data[NUMERIC_COLS + ["churned"]].corr()["churned"].drop("churned")
    print(corr.sort_values(ascending=False).round(3).to_string())

    print("\n--- Average profile: churned vs retained ---")
    print(data.groupby("churned")[NUMERIC_COLS].mean().round(2).T.to_string())


# ----------------------------------------------------------------------------
# 4. Visualize
# ----------------------------------------------------------------------------

def plot_overview(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(data=data, x="age", hue="churned", kde=True, bins=20, ax=axes[0, 0])
    axes[0, 0].set_title("Customer Age Distribution by Churn")
    axes[0, 0].set_xlabel("Age")

    sns.boxplot(data=data, x="churned", y="days_since_last_visit", ax=axes[0, 1])
    axes[0, 1].set_title("Days Since Last Visit: Retained (0) vs Churned (1)")
    axes[0, 1].set_xlabel("Churned")
    axes[0, 1].set_ylabel("Days Since Last Visit")

    sns.barplot(data=data, x="loyalty_member", y="churned", errorbar=None, ax=axes[1, 0])
    axes[1, 0].set_title("Churn Rate by Loyalty Membership")
    axes[1, 0].set_xlabel("Loyalty Member")
    axes[1, 0].set_ylabel("Churn Rate")
    axes[1, 0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    corr = data[NUMERIC_COLS + ["churned"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, annot_kws={"size": 8}, ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Matrix")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_overview.png", dpi=150)
    plt.close(fig)


def plot_churn_drivers(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    percent = plt.matplotlib.ticker.PercentFormatter(1.0)

    for ax, col, title in [
        (axes[0], "recency_group", "Days Since Last Visit"),
        (axes[1], "visits_last_3_months", "Visits in Last 3 Months"),
        (axes[2], "late_deliveries", "Late Deliveries"),
    ]:
        sns.barplot(data=data, x=col, y="churned", errorbar=None, ax=ax, color="steelblue")
        ax.axhline(data["churned"].mean(), color="red", linestyle="--", label="Overall churn")
        ax.set_title(f"Churn Rate by {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Churn Rate")
        ax.yaxis.set_major_formatter(percent)
        ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_drivers.png", dpi=150)
    plt.close(fig)


def plot_missing_values(raw):
    fig, ax = plt.subplots(figsize=(10, 4))
    missing = raw.isnull().sum().sort_values(ascending=False)
    sns.barplot(x=missing.index, y=missing.values, ax=ax, color="indianred")
    ax.set_title("Missing Values per Column (Raw Data)")
    ax.set_ylabel("Missing values")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_values.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Prepare for modelling
# ----------------------------------------------------------------------------

def save_model_ready(data):
    section("4. MODEL-READY DATASET")
    model_data = data.drop(columns=["customer_id", "recency_group", "age_group"])
    model_data["loyalty_member"] = (model_data["loyalty_member"] == "Yes").astype(int)
    model_data = pd.get_dummies(model_data, columns=["gender", "city"], drop_first=True, dtype=int)

    out_path = PROCESSED_DIR / "shop_customers_clean.csv"
    model_data.to_csv(out_path, index=False)
    print(model_data.info())
    print(f"\nSaved model-ready dataset to {out_path}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    raw = load_data(RAW_DATA)
    plot_missing_values(raw)

    data = clean_data(raw.copy())
    data = add_segments(data)
    analyse(data)

    plot_overview(data)
    plot_churn_drivers(data)
    print(f"\nFigures saved to {FIGURES_DIR}/")

    save_model_ready(data)


if __name__ == "__main__":
    main()
