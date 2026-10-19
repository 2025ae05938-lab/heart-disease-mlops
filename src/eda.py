"""Step 1b: Exploratory Data Analysis. Saves plots into screenshots/eda/.

Run from the project root:
    python src/eda.py
You can copy these lines into a Jupyter notebook cell by cell as well.
"""
import os

import matplotlib

matplotlib.use("Agg")  # draw to files, no pop-up window needed
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from data_utils import (  # noqa: E402
    NUMERIC_FEATURES,
    TARGET,
    load_clean,
    save_clean,
)

OUT_DIR = os.path.join("screenshots", "eda")  # where plots are saved


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_clean()  # load + clean
    save_clean(df)  # also save data/heart_clean.csv
    print("Shape:", df.shape)
    print(df.describe().T)  # summary statistics per column

    # Plot 1: histograms of the numeric features
    df[NUMERIC_FEATURES].hist(bins=20, figsize=(12, 7))
    plt.suptitle("Distribution of numeric features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "histograms.png"), dpi=150)
    plt.close()

    # Plot 2: correlation heatmap (how strongly columns move together)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Plot 3: class balance (are healthy and diseased patients equally common?)
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df)
    plt.title("Class balance (0 = no disease, 1 = disease)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_balance.png"), dpi=150)
    plt.close()

    # Plot 4 (extra): numeric features split by target, to see which separate classes
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "boxplots_by_target.png"), dpi=150)
    plt.close()

    print(f"Plots saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
