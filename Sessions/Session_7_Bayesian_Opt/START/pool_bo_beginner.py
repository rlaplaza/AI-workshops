"""
Beginner-friendly pool-based Bayesian optimization demo.

This script is deliberately sequential and very verbose:
- It prints many pandas snippets so you can track every step.
- It creates a candidate pool by combining observed entries and scanning
  around temperature / loading / Selectfluor values.
- It compares two uncertainty-aware models:
  1) Gaussian Process Regressor
  2) Random Forest with tree-ensemble uncertainty proxy
- It runs a simple expected-improvement BO loop over the candidate pool.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------- User settings ------------------------------- #
CSV_PATH = Path(__file__).with_name("Pd_fluorination_optimization_START_example.csv")
TARGET_COL = "yield"
N_BO_STEPS = 8
RANDOM_STATE = 7
BATCH_SIZE = 1

# Exploration settings for candidate generation.
TEMPERATURE_SCAN = np.arange(40, 101, 10)  # 40, 50, ..., 100
PD_LOADING_SCAN = np.array([5, 10, 15])
SELECTFLUOR_SCAN = np.array([1.0, 1.5, 2.0, 2.5])


def print_banner(title: str) -> None:
    """Pretty section separator."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def show_current_figure() -> None:
    """Show then close figure to avoid many-open-figure warnings."""
    plt.tight_layout()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
        )
        plt.show()
    plt.close()


def gaussian_process_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Pipeline with preprocessing + Gaussian Process."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    # Keep GP intentionally simple and robust for this tiny beginner dataset.
    # A single isotropic length scale is often easier to fit than high-dim ARD.
    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * Matern(length_scale=1.0, length_scale_bounds=(0.2, 5.0), nu=1.5)
        + WhiteKernel(noise_level=0.2, noise_level_bounds=(0.01, 2.0))
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-3,
        random_state=RANDOM_STATE,
        n_restarts_optimizer=6,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def random_forest_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Pipeline with preprocessing + Random Forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=RANDOM_STATE,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def quick_inspect_csv(df: pd.DataFrame) -> None:
    """Print a short, beginner-friendly snapshot of the CSV."""
    print_banner("STEP 1A - QUICK CSV INSPECTION")
    n_rows, n_cols = df.shape
    print(f"Rows: {n_rows} | Columns: {n_cols}")

    print("\nFirst 5 rows:")
    print(df.head(5))

    print("\nMissing values per column:")
    print(df.isna().sum())

    if TARGET_COL in df.columns:
        print(f"\n{TARGET_COL} summary:")
        print(df[TARGET_COL].describe())

    object_cols = list(df.select_dtypes(include=["object"]).columns)
    if object_cols:
        print("\nUnique value counts for categorical columns:")
        for col in object_cols:
            print(f"  - {col}: {df[col].nunique()}")


def ask_user_for_observed_yield(
    chosen_features: pd.DataFrame,
    step: int,
    batch_rank: int,
    suggested_yield: float,
) -> float:
    """
    Ask the user for an observed yield for the selected condition.

    This is strict manual mode: input is required and empty responses are rejected.
    """
    print("\nSelected condition for lab execution:")
    print(chosen_features)
    print(
        f"Model-suggested yield is {suggested_yield:.3f}. "
        "Enter your measured yield (required; no default accepted)."
    )

    while True:
        user_text = input(f"Step {step}, batch rank {batch_rank} measured yield: ").strip()
        if user_text == "":
            print("Input is required. Please enter a numeric measured yield.")
            continue
        try:
            measured_value = float(user_text)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if measured_value < 0 or measured_value > 100:
            print("Yield is usually between 0 and 100. Please re-enter.")
            continue
        return measured_value


def rf_bootstrap_predict_mean_std(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_pool: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit two bootstrapped RF models and combine all trees for uncertainty.

    This keeps RF uncertainty simple for beginners:
    - train RF #1 on bootstrap sample #1
    - train RF #2 on bootstrap sample #2
    - collect predictions from every tree in both forests
    - use tree-level mean and std as prediction + uncertainty proxy
    """
    n_rows = len(x_train)
    rng = np.random.RandomState(RANDOM_STATE)
    all_tree_predictions = []

    for bootstrap_id in range(2):
        sampled_idx = rng.choice(n_rows, size=n_rows, replace=True)
        x_boot = x_train.iloc[sampled_idx].reset_index(drop=True)
        y_boot = y_train.iloc[sampled_idx].reset_index(drop=True)

        rf_pipe = random_forest_pipeline(cat_cols, num_cols)
        rf_pipe.fit(x_boot, y_boot)

        x_pool_encoded = rf_pipe.named_steps["prep"].transform(x_pool)
        rf_model = rf_pipe.named_steps["model"]
        for tree in rf_model.estimators_:
            tree_predictions = tree.predict(x_pool_encoded)
            all_tree_predictions.append(tree_predictions)

        print(
            f"[RF diagnostics] finished bootstrap forest {bootstrap_id + 1}/2 "
            f"with {len(rf_model.estimators_)} trees"
        )

    tree_pred_matrix = np.vstack(all_tree_predictions)
    mean_pred = tree_pred_matrix.mean(axis=0)
    std_pred = tree_pred_matrix.std(axis=0, ddof=1)
    return mean_pred, std_pred


def fit_predict_with_uncertainty(
    model_name: str,
    pipe: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_pool: pd.DataFrame,
    print_diagnostics: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit model and return mean + uncertainty on pool.

    For GPR:
      - uncertainty is predictive standard deviation from GP.
    For RF:
      - uncertainty proxy is std across trees from two bootstrapped forests.
    """
    if model_name == "gpr":
        # Keep output beginner-friendly by silencing optimizer internals.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipe.fit(x_train, y_train)
        mean_pred, std_pred = pipe.predict(x_pool, return_std=True)
        if print_diagnostics:
            gp_model = pipe.named_steps["model"]
            print(
                "[GPR diagnostics] "
                f"kernel={gp_model.kernel_} | "
                f"mu_std={np.std(mean_pred):.3f} | "
                f"mu_range=({np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}) | "
                f"sd_range=({np.min(std_pred):.3f}, {np.max(std_pred):.3f})"
            )
        return mean_pred, std_pred

    if model_name == "rf":
        # Keep this branch explicit: unpack feature groups from the pipeline and
        # call a helper that trains two bootstrapped forests.
        preprocessor = pipe.named_steps["prep"]
        transformers = preprocessor.transformers
        cat_cols = list(transformers[0][2])
        num_cols = list(transformers[1][2])
        mean_pred, std_pred = rf_bootstrap_predict_mean_std(
            x_train=x_train,
            y_train=y_train,
            x_pool=x_pool,
            cat_cols=cat_cols,
            num_cols=num_cols,
        )
        return mean_pred, std_pred

    raise ValueError(f"Unknown model_name={model_name}")


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    """Expected improvement for maximization."""
    sigma = np.clip(sigma, 1e-12, None)
    improvement = mu - best_y - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma <= 1e-12, 0.0, ei)


def plot_iteration_diagnostics(
    ranking: pd.DataFrame,
    selected_batch: pd.DataFrame,
    feature_cols: list[str],
    step: int,
    best_so_far: float,
    batch_size: int,
) -> None:
    """Visualize BO state for one batch iteration."""
    selected_keys = set(map(tuple, selected_batch[feature_cols].to_numpy()))
    ranking_plot = ranking.copy()
    ranking_plot["is_selected_batch"] = ranking_plot[feature_cols].apply(tuple, axis=1).isin(selected_keys)

    # Plot 1: Predicted mean vs uncertainty, colored by EI.
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        ranking_plot.loc[~ranking_plot["is_selected_batch"], "pred_std"],
        ranking_plot.loc[~ranking_plot["is_selected_batch"], "pred_mean"],
        c=ranking_plot.loc[~ranking_plot["is_selected_batch"], "EI"],
        cmap="viridis",
        alpha=0.45,
        s=28,
    )
    plt.scatter(
        ranking_plot.loc[ranking_plot["is_selected_batch"], "pred_std"],
        ranking_plot.loc[ranking_plot["is_selected_batch"], "pred_mean"],
        color="red",
        edgecolor="black",
        s=75,
        zorder=5,
        label=f"Selected batch (top {batch_size} EI)",
    )
    plt.colorbar(sc, label="Expected Improvement (EI)")
    plt.axhline(best_so_far, linestyle="--", linewidth=1, color="gray", label="Current best observed")
    plt.xlabel("Predicted uncertainty (std)")
    plt.ylabel("Predicted mean yield")
    plt.title(f"Iteration {step}: uncertainty vs predicted mean (batch-highlighted)")
    plt.legend()
    show_current_figure()

    # Plot 2: EI for top candidates, highlighting selected batch members.
    top_k = ranking.head(max(20, batch_size)).copy()
    top_k_keys = top_k[feature_cols].apply(tuple, axis=1)
    top_k_selected = top_k_keys.isin(selected_keys)
    labels = []
    label_cols = feature_cols[:4]
    for _, row in top_k.iterrows():
        label_parts = [f"{col}={row[col]}" for col in label_cols]
        labels.append(" | ".join(label_parts))
    bar_colors = ["tomato" if is_sel else "steelblue" for is_sel in top_k_selected]
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(top_k)), top_k["EI"], color=bar_colors)
    plt.xticks(range(len(top_k)), labels, rotation=45, ha="right")
    plt.ylabel("Expected Improvement (EI)")
    plt.xlabel("Top candidates (highest EI on left)")
    plt.title(f"Iteration {step}: top candidates by EI (selected batch in red)")
    show_current_figure()

    # Quick textual reminder of selected batch settings.
    batch_preview_cols = feature_cols + ["pred_mean", "pred_std", "EI"]
    selected_preview = selected_batch[batch_preview_cols].copy().reset_index(drop=True)
    selected_preview.insert(0, "batch_rank", np.arange(1, len(selected_preview) + 1))
    print(f"[Plot note] Selected batch for iteration {step}:")
    print(selected_preview)


def main() -> None:
    np.random.seed(RANDOM_STATE)

    # 1) Load and inspect the data.
    print_banner("STEP 1 - LOAD DATA")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV from: {CSV_PATH}")
    quick_inspect_csv(df)

    print_banner("STEP 1B - FULL DATA INSPECTION")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nNumerical summary:")
    print(df.describe(include=[np.number]))
    print("\nCategorical summary:")
    print(df.describe(include=["object"]))

    # 2) Define columns.
    print_banner("STEP 2 - DEFINE FEATURE COLUMNS")
    feature_cols = []
    for col in df.columns:
        if col != TARGET_COL:
            feature_cols.append(col)

    cat_cols = []
    num_cols = []
    for col in feature_cols:
        if df[col].dtype == "object":
            cat_cols.append(col)
        else:
            num_cols.append(col)
    print(f"Target column: {TARGET_COL}")
    print(f"Feature columns: {feature_cols}")
    print(f"Categorical columns: {cat_cols}")
    print(f"Numeric columns: {num_cols}")

    # 3) Build candidate pool:
    #    combine all observed categorical combinations with scanned numeric ranges.
    print_banner("STEP 3 - BUILD CANDIDATE POOL (OBSERVED + SCANNED CONDITIONS)")
    unique_cat = df[cat_cols].drop_duplicates().reset_index(drop=True)
    print("Unique categorical combinations (first 10):")
    print(unique_cat.head(10))
    print(f"Count of unique categorical combinations: {len(unique_cat)}")

    numeric_grid = pd.MultiIndex.from_product(
        [TEMPERATURE_SCAN, PD_LOADING_SCAN, SELECTFLUOR_SCAN],
        names=["Temperature", "Pd_loading", "Selectfluor"],
    ).to_frame(index=False)
    print("\nNumeric scan grid (first 10):")
    print(numeric_grid.head(10))
    print(f"Count of numeric grid rows: {len(numeric_grid)}")

    # Cross join unique categorical choices with scanned numeric values.
    pool = unique_cat.merge(numeric_grid, how="cross")
    pool = pool[feature_cols].copy()
    print("\nCandidate pool from cross join (first 10):")
    print(pool.head(10))
    print(f"Candidate pool size before dedup: {len(pool)}")
    pool = pool.drop_duplicates().reset_index(drop=True)
    print(f"Candidate pool size after dedup: {len(pool)}")

    # Mark which points are already observed in the original dataset.
    observed_features = df[feature_cols].drop_duplicates()
    observed_keys = set(map(tuple, observed_features.to_numpy()))
    pool["_is_observed"] = [tuple(row) in observed_keys for row in pool[feature_cols].to_numpy()]
    print("\nObserved-vs-unobserved count in pool:")
    print(pool["_is_observed"].value_counts(dropna=False))
    print("\nRandom pool sample:")
    print(pool.sample(min(10, len(pool)), random_state=RANDOM_STATE))

    # 4) Split into training and candidate pool for BO.
    print_banner("STEP 4 - TRAINING DATA AND AVAILABLE (UNOBSERVED) CANDIDATES")
    observed_df = df.copy()
    candidate_df = pool.loc[~pool["_is_observed"], feature_cols].copy()
    print(f"Observed experiments available for training: {len(observed_df)}")
    print(f"Unobserved candidates available for BO: {len(candidate_df)}")
    print("\nObserved training rows (head):")
    print(observed_df.head())
    print("\nCandidate rows (head):")
    print(candidate_df.head())

    # 5) Compare two uncertainty-aware models on same pool before BO loop.
    print_banner("STEP 5 - MODEL DIAGNOSTIC: GPR VS RF UNCERTAINTY")
    x_train = observed_df[feature_cols]
    y_train = observed_df[TARGET_COL]

    gpr_pipe = gaussian_process_pipeline(cat_cols, num_cols)
    rf_pipe = random_forest_pipeline(cat_cols, num_cols)

    mu_gpr, sd_gpr = fit_predict_with_uncertainty("gpr", gpr_pipe, x_train, y_train, candidate_df)
    mu_rf, sd_rf = fit_predict_with_uncertainty("rf", rf_pipe, x_train, y_train, candidate_df)

    diagnostic = candidate_df.copy()
    diagnostic["mu_gpr"] = mu_gpr
    diagnostic["sd_gpr"] = sd_gpr
    diagnostic["mu_rf"] = mu_rf
    diagnostic["sd_rf"] = sd_rf
    diagnostic["abs_mu_diff"] = np.abs(diagnostic["mu_gpr"] - diagnostic["mu_rf"])
    diagnostic["abs_sd_diff"] = np.abs(diagnostic["sd_gpr"] - diagnostic["sd_rf"])

    print("Top 10 points where model means disagree the most:")
    print(diagnostic.sort_values("abs_mu_diff", ascending=False).head(10))
    print("\nTop 10 points where uncertainty estimates disagree the most:")
    print(diagnostic.sort_values("abs_sd_diff", ascending=False).head(10))

    plt.figure(figsize=(8, 6))
    plt.scatter(diagnostic["mu_gpr"], diagnostic["mu_rf"], alpha=0.5)
    low = min(diagnostic["mu_gpr"].min(), diagnostic["mu_rf"].min())
    high = max(diagnostic["mu_gpr"].max(), diagnostic["mu_rf"].max())
    plt.plot([low, high], [low, high], "--", linewidth=1)
    plt.xlabel("GPR predicted mean yield")
    plt.ylabel("RF predicted mean yield")
    plt.title("Model diagnostic: predicted mean comparison")
    show_current_figure()

    plt.figure(figsize=(8, 6))
    plt.scatter(diagnostic["sd_gpr"], diagnostic["sd_rf"], alpha=0.5)
    low = min(diagnostic["sd_gpr"].min(), diagnostic["sd_rf"].min())
    high = max(diagnostic["sd_gpr"].max(), diagnostic["sd_rf"].max())
    plt.plot([low, high], [low, high], "--", linewidth=1)
    plt.xlabel("GPR uncertainty (std)")
    plt.ylabel("RF uncertainty proxy (std across trees)")
    plt.title("Model diagnostic: uncertainty comparison")
    show_current_figure()

    # 5B) Minimal 5-fold CV on observed data to compare test-fold predictions.
    print_banner("STEP 5B - 5-FOLD CV: TEST PREDICTION COMPARISON (GPR VS RF)")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_true_all: list[float] = []
    y_pred_gpr_all: list[float] = []
    y_pred_rf_all: list[float] = []
    y_std_gpr_all: list[float] = []
    y_std_rf_all: list[float] = []
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(observed_df), start=1):
        train_fold = observed_df.iloc[train_idx].reset_index(drop=True)
        test_fold = observed_df.iloc[test_idx].reset_index(drop=True)

        x_train_fold = train_fold[feature_cols]
        y_train_fold = train_fold[TARGET_COL]
        x_test_fold = test_fold[feature_cols]
        y_test_fold = test_fold[TARGET_COL].to_numpy(dtype=float)

        gpr_fold = gaussian_process_pipeline(cat_cols, num_cols)
        rf_fold = random_forest_pipeline(cat_cols, num_cols)
        pred_gpr_fold, std_gpr_fold = fit_predict_with_uncertainty(
            "gpr", gpr_fold, x_train_fold, y_train_fold, x_test_fold, print_diagnostics=False
        )
        pred_rf_fold, std_rf_fold = fit_predict_with_uncertainty(
            "rf", rf_fold, x_train_fold, y_train_fold, x_test_fold, print_diagnostics=False
        )
        pred_gpr_fold = pred_gpr_fold.astype(float)
        pred_rf_fold = pred_rf_fold.astype(float)
        std_gpr_fold = std_gpr_fold.astype(float)
        std_rf_fold = std_rf_fold.astype(float)

        y_true_all.extend(y_test_fold.tolist())
        y_pred_gpr_all.extend(pred_gpr_fold.tolist())
        y_pred_rf_all.extend(pred_rf_fold.tolist())
        y_std_gpr_all.extend(std_gpr_fold.tolist())
        y_std_rf_all.extend(std_rf_fold.tolist())

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_test": len(test_fold),
                "mae_gpr": mean_absolute_error(y_test_fold, pred_gpr_fold),
                "mae_rf": mean_absolute_error(y_test_fold, pred_rf_fold),
                "r2_gpr": r2_score(y_test_fold, pred_gpr_fold),
                "r2_rf": r2_score(y_test_fold, pred_rf_fold),
            }
        )

    cv_df = pd.DataFrame(fold_rows)
    print("Per-fold test metrics:")
    print(cv_df)
    print("\nMean CV metrics:")
    print(cv_df[["mae_gpr", "mae_rf", "r2_gpr", "r2_rf"]].mean().to_frame("mean"))

    cv_pred_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true_all, dtype=float),
            "y_pred_gpr": np.asarray(y_pred_gpr_all, dtype=float),
            "y_pred_rf": np.asarray(y_pred_rf_all, dtype=float),
            "std_gpr": np.asarray(y_std_gpr_all, dtype=float),
            "std_rf": np.asarray(y_std_rf_all, dtype=float),
        }
    )
    cv_pred_df["abs_err_gpr"] = np.abs(cv_pred_df["y_true"] - cv_pred_df["y_pred_gpr"])
    cv_pred_df["abs_err_rf"] = np.abs(cv_pred_df["y_true"] - cv_pred_df["y_pred_rf"])

    gpr_pearson_r, gpr_pearson_p = pearsonr(cv_pred_df["std_gpr"], cv_pred_df["abs_err_gpr"])
    gpr_spearman_rho, gpr_spearman_p = spearmanr(cv_pred_df["std_gpr"], cv_pred_df["abs_err_gpr"])
    rf_pearson_r, rf_pearson_p = pearsonr(cv_pred_df["std_rf"], cv_pred_df["abs_err_rf"])
    rf_spearman_rho, rf_spearman_p = spearmanr(cv_pred_df["std_rf"], cv_pred_df["abs_err_rf"])

    print("\nUncertainty vs error correlation on test folds:")
    print(
        f"GPR: Pearson r={gpr_pearson_r:.3f} (p={gpr_pearson_p:.2e}), "
        f"Spearman rho={gpr_spearman_rho:.3f} (p={gpr_spearman_p:.2e})"
    )
    print(
        f"RF : Pearson r={rf_pearson_r:.3f} (p={rf_pearson_p:.2e}), "
        f"Spearman rho={rf_spearman_rho:.3f} (p={rf_spearman_p:.2e})"
    )

    cv_pred_df["uncertainty_bin"] = pd.qcut(cv_pred_df["std_gpr"], q=4, duplicates="drop")
    gpr_bin_summary = (
        cv_pred_df.groupby("uncertainty_bin", observed=True)
        .agg(n_points=("abs_err_gpr", "size"), mean_pred_std=("std_gpr", "mean"), mean_abs_err=("abs_err_gpr", "mean"))
        .reset_index()
    )
    cv_pred_df["uncertainty_bin"] = pd.qcut(cv_pred_df["std_rf"], q=4, duplicates="drop")
    rf_bin_summary = (
        cv_pred_df.groupby("uncertainty_bin", observed=True)
        .agg(n_points=("abs_err_rf", "size"), mean_pred_std=("std_rf", "mean"), mean_abs_err=("abs_err_rf", "mean"))
        .reset_index()
    )
    print("\nGPR uncertainty-bin summary (higher mean_pred_std should trend to higher mean_abs_err):")
    print(gpr_bin_summary)
    print("\nRF uncertainty-bin summary (higher mean_pred_std should trend to higher mean_abs_err):")
    print(rf_bin_summary)

    print("\nTop 10 test points where model predictions disagree most:")
    print(
        cv_pred_df.assign(pred_disagreement=np.abs(cv_pred_df["y_pred_gpr"] - cv_pred_df["y_pred_rf"]))
        .sort_values("pred_disagreement", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(cv_pred_df["y_true"], cv_pred_df["y_pred_gpr"], alpha=0.7, label="GPR")
    plt.scatter(cv_pred_df["y_true"], cv_pred_df["y_pred_rf"], alpha=0.7, label="RF")
    low = min(cv_pred_df["y_true"].min(), cv_pred_df["y_pred_gpr"].min(), cv_pred_df["y_pred_rf"].min())
    high = max(cv_pred_df["y_true"].max(), cv_pred_df["y_pred_gpr"].max(), cv_pred_df["y_pred_rf"].max())
    plt.plot([low, high], [low, high], "--", linewidth=1, color="black")
    plt.xlabel("True yield (test folds)")
    plt.ylabel("Predicted yield")
    plt.title("5-fold CV test predictions: GPR vs RF")
    plt.legend()
    show_current_figure()

    plt.figure(figsize=(8, 6))
    plt.scatter(cv_pred_df["abs_err_gpr"], cv_pred_df["abs_err_rf"], alpha=0.7)
    low = min(cv_pred_df["abs_err_gpr"].min(), cv_pred_df["abs_err_rf"].min())
    high = max(cv_pred_df["abs_err_gpr"].max(), cv_pred_df["abs_err_rf"].max())
    plt.plot([low, high], [low, high], "--", linewidth=1, color="black")
    plt.xlabel("Absolute error GPR (test folds)")
    plt.ylabel("Absolute error RF (test folds)")
    plt.title("5-fold CV absolute error comparison")
    show_current_figure()

    plt.figure(figsize=(8, 6))
    plt.scatter(cv_pred_df["std_gpr"], cv_pred_df["abs_err_gpr"], alpha=0.7, label="GPR")
    plt.scatter(cv_pred_df["std_rf"], cv_pred_df["abs_err_rf"], alpha=0.7, label="RF")
    plt.xlabel("Predicted uncertainty (std, test folds)")
    plt.ylabel("Absolute error (test folds)")
    plt.title("5-fold CV: does uncertainty track error?")
    plt.legend()
    show_current_figure()

    # 6) Sequential BO flow (teaching style) with batched selection.
    print_banner("STEP 6 - BATCHED SEQUENTIAL BO")
    bo_train = observed_df.copy()
    bo_pool = candidate_df.copy()
    selected_history = []

    # ---------------------------------------------------------------------
    # STEP 6A - Define helper functions for batched BO.
    # ---------------------------------------------------------------------
    print_banner("STEP 6A - DEFINE HELPER FUNCTIONS FOR BATCHED ITERATIONS")
    print(
        "\nHelper roles:\n"
        "- build_ranking_table: fit model + compute EI + sort candidates\n"
        "- choose_batch_and_record: select top-k EI candidates + collect observed yields\n"
        "- update_train_and_pool: append batch and remove selected conditions from pool"
    )

    def build_ranking_table(current_train: pd.DataFrame, current_pool: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Return ranked candidate table and current best observed yield."""
        x_train_local = current_train[feature_cols]
        y_train_local = current_train[TARGET_COL]
        gpr_local = gaussian_process_pipeline(cat_cols, num_cols)
        mu_local, sd_local = fit_predict_with_uncertainty(
            "gpr", gpr_local, x_train_local, y_train_local, current_pool
        )
        best_local = float(y_train_local.max())
        ei_local = expected_improvement(mu_local, sd_local, best_local, xi=0.01)

        ranking_local = current_pool.copy()
        ranking_local["pred_mean"] = mu_local
        ranking_local["pred_std"] = sd_local
        ranking_local["EI"] = ei_local
        ranking_local = ranking_local.sort_values("EI", ascending=False).reset_index(drop=True)
        return ranking_local, best_local

    def choose_batch_and_record(
        ranking_local: pd.DataFrame, step_local: int, batch_size_local: int
    ) -> tuple[pd.DataFrame, list[tuple], list[dict], pd.DataFrame]:
        """Select top-k candidates and create observed-yield records."""
        batch = ranking_local.head(batch_size_local).copy().reset_index(drop=True)
        chosen_records_local = []
        chosen_keys_local = []
        history_items_local = []

        for batch_rank, (_, chosen_local) in enumerate(batch.iterrows(), start=1):
            chosen_features_local = chosen_local[feature_cols].to_frame().T
            chosen_key_local = tuple(chosen_features_local.iloc[0][feature_cols].to_list())
            observed_yield_local = ask_user_for_observed_yield(
                chosen_features=chosen_features_local,
                step=step_local,
                batch_rank=batch_rank,
                suggested_yield=float(chosen_local["pred_mean"]),
            )

            chosen_record_local = chosen_features_local.copy()
            chosen_record_local[TARGET_COL] = observed_yield_local
            chosen_record_local["source"] = f"bo_step_{step_local}_batch_{batch_rank}_manual"
            chosen_records_local.append(chosen_record_local)
            chosen_keys_local.append(chosen_key_local)

            history_item_local = {
                "step": step_local,
                "batch_rank": batch_rank,
                "best_before": np.nan,  # Filled by caller.
                "pred_mean": float(chosen_local["pred_mean"]),
                "pred_std": float(chosen_local["pred_std"]),
                "EI": float(chosen_local["EI"]),
                TARGET_COL: observed_yield_local,
                **{col: chosen_local[col] for col in feature_cols},
            }
            history_items_local.append(history_item_local)

        chosen_records_df = pd.concat(chosen_records_local, ignore_index=True)
        return chosen_records_df, chosen_keys_local, history_items_local, batch

    def update_train_and_pool(
        current_train: pd.DataFrame,
        current_pool: pd.DataFrame,
        chosen_record_local: pd.DataFrame,
        chosen_keys_local: list[tuple],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Append chosen batch to train and remove all selected keys from pool."""
        new_train = pd.concat([current_train, chosen_record_local], ignore_index=True)
        pool_keys_local = current_pool[feature_cols].apply(tuple, axis=1)
        chosen_key_set = set(chosen_keys_local)
        new_pool = current_pool.loc[~pool_keys_local.isin(chosen_key_set)].reset_index(drop=True)
        return new_train, new_pool

    # ---------------------------------------------------------------------
    # STEP 6B - Batched BO loop.
    # ---------------------------------------------------------------------
    print_banner("STEP 6B - BATCHED BO LOOP")
    for step in range(1, N_BO_STEPS + 1):
        if len(bo_pool) == 0:
            print("Pool is empty; stopping early.")
            break
        ranking, best_so_far = build_ranking_table(bo_train, bo_pool)
        current_batch_size = min(BATCH_SIZE, len(ranking))

        print_banner(f"BO ITERATION {step}")
        print(f"Current best observed yield: {best_so_far:.3f}")
        print(f"Batch size this iteration: {current_batch_size}")
        print("Top 5 candidates by EI:")
        print(ranking.head(5))

        chosen_records, chosen_keys, history_items, selected_batch = choose_batch_and_record(
            ranking, step, current_batch_size
        )
        for history_item in history_items:
            history_item["best_before"] = best_so_far

        plot_iteration_diagnostics(
            ranking=ranking,
            selected_batch=selected_batch,
            feature_cols=feature_cols,
            step=step,
            best_so_far=best_so_far,
            batch_size=current_batch_size,
        )

        batch_yields = [float(item[TARGET_COL]) for item in history_items]
        print(
            f"Batch predicted means range: {float(ranking.head(current_batch_size)['pred_mean'].min()):.3f} "
            f"to {float(ranking.head(current_batch_size)['pred_mean'].max()):.3f}"
        )
        print(
            f"Revealed observed yields range: {min(batch_yields):.3f} to {max(batch_yields):.3f}"
        )
        print("Per-candidate prediction vs observation:")
        for item in history_items:
            pred_mean_local = float(item["pred_mean"])
            pred_std_local = float(item["pred_std"])
            obs_local = float(item[TARGET_COL])
            print(
                f"  - rank {int(item['batch_rank'])}: "
                f"pred={pred_mean_local:.3f} +/- {pred_std_local:.3f}, "
                f"observed={obs_local:.3f}"
            )

        selected_history.extend(history_items)
        bo_train, bo_pool = update_train_and_pool(bo_train, bo_pool, chosen_records, chosen_keys)

        print(f"Training set size is now: {len(bo_train)}")
        print(f"Pool size is now: {len(bo_pool)}")

    print(
        "\nRecap after batched loop:\n"
        "- At each iteration, the model ranked pool points by EI.\n"
        f"- We selected top {BATCH_SIZE} points as a batch (or fewer if pool was smaller).\n"
        "- Observations were entered after selection and then appended to training."
    )

    history_df = pd.DataFrame(selected_history)
    print_banner("STEP 7 - BO HISTORY TABLES")
    print("Full BO history:")
    print(history_df)
    print("\nYield progression (best_before vs chosen observed yield):")
    print(history_df[["step", "best_before", TARGET_COL, "pred_std", "EI"]])

    step_summary = (
        history_df.groupby("step", as_index=False)
        .agg(
            best_before=("best_before", "first"),
            observed_mean=(TARGET_COL, "mean"),
            observed_std=(TARGET_COL, "std"),
            ei_mean=("EI", "mean"),
            ei_std=("EI", "std"),
            pred_std_mean=("pred_std", "mean"),
            pred_std_std=("pred_std", "std"),
            n_points=("EI", "size"),
        )
        .sort_values("step")
        .reset_index(drop=True)
    )
    step_summary["observed_std"] = step_summary["observed_std"].fillna(0.0)
    step_summary["ei_std"] = step_summary["ei_std"].fillna(0.0)
    step_summary["pred_std_std"] = step_summary["pred_std_std"].fillna(0.0)

    plt.figure(figsize=(8, 5))
    plt.plot(step_summary["step"], step_summary["best_before"], marker="o", label="Best observed before step")
    plt.errorbar(
        step_summary["step"],
        step_summary["observed_mean"],
        yerr=step_summary["observed_std"],
        fmt="s-",
        capsize=4,
        label="Observed yield (mean +/- std within batch)",
    )
    plt.xlabel("BO step")
    plt.ylabel("Yield")
    plt.title("Sequential BO trajectory (batch-aware)")
    plt.legend()
    show_current_figure()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        step_summary["step"],
        step_summary["ei_mean"],
        yerr=step_summary["ei_std"],
        fmt="o-",
        capsize=4,
        label="EI (mean +/- std within batch)",
    )
    plt.xlabel("BO step")
    plt.ylabel("Expected Improvement")
    plt.title("Acquisition value per BO step (batch-aware)")
    plt.legend()
    show_current_figure()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        step_summary["step"],
        step_summary["pred_std_mean"],
        yerr=step_summary["pred_std_std"],
        fmt="o-",
        capsize=4,
        label="Predicted std (mean +/- std within batch)",
    )
    plt.xlabel("BO step")
    plt.ylabel("Predicted uncertainty (std)")
    plt.title("Uncertainty per BO step (batch-aware)")
    plt.legend()
    show_current_figure()

    print_banner("DONE")
    print("Script completed successfully.")
    print("Tip: enter measured lab yields for each selected batch member in a human-in-the-loop BO campaign.")


if __name__ == "__main__":
    main()
