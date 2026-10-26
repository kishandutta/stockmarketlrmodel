"""
Next-Day Stock Market Close Price Forecasting with Scikit-Learn Linear Regression

This script trains a Scikit-Learn Linear Regression model to forecast the next day's
Close price using historical market data (OHLC, Volume, Lags, Moving Averages, and Indices).
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set aesthetic style for matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def load_and_preprocess_data(csv_path: str, target_index: str = "all") -> pd.DataFrame:
    """
    Load dataset, clean missing values, parse dates, and sort chronologically.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file '{csv_path}' not found.")

    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Filter by specific index if requested
    if target_index and target_index.lower() != "all":
        valid_indices = df["Index"].unique()
        if target_index not in valid_indices:
            raise ValueError(
                f"Index '{target_index}' not found in dataset. Available: {list(valid_indices)}"
            )
        df = df[df["Index"] == target_index].copy()
        print(f"Filtered dataset for Index: {target_index} ({len(df):,} records)")
    else:
        print(f"Loaded {len(df):,} records across {df['Index'].nunique()} market indices.")

    # Drop records missing core OHLC prices
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # Parse and sort by Index and Date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Index", "Date"]).reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct time-series lag and technical features grouped by market index.
    Features include:
    - Target: Next day's Close price (Close shifted by -1)
    - Lagged prices (Close Lag 1, Lag 2, Lag 5)
    - Moving averages (SMA 5, SMA 10)
    - Intraday price returns & volatility
    - Log Volume
    """
    print("Engineering features (lags, rolling averages, returns)...")
    grouped = df.groupby("Index")

    # Target: Next Day Close
    df["Target_Next_Close"] = grouped["Close"].shift(-1)

    # Lags (Prior days' Close)
    df["Close_Lag1"] = grouped["Close"].shift(1)
    df["Close_Lag2"] = grouped["Close"].shift(2)
    df["Close_Lag5"] = grouped["Close"].shift(5)

    # Rolling Simple Moving Averages
    df["SMA_5"] = grouped["Close"].transform(lambda s: s.rolling(window=5).mean())
    df["SMA_10"] = grouped["Close"].transform(lambda s: s.rolling(window=10).mean())

    # Intraday returns and range
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]

    # Log Volume (handle 0 volume safely)
    volume_series = df["Volume"].fillna(0).clip(lower=0)
    df["Log_Volume"] = np.log1p(volume_series)

    # Drop rows that have NaNs due to rolling/shifting
    initial_rows = len(df)
    clean_df = df.dropna().copy().reset_index(drop=True)
    dropped = initial_rows - len(clean_df)
    print(f"Cleaned feature dataset: {len(clean_df):,} rows ({dropped:,} boundary/null rows dropped).")

    return clean_df


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.20):
    """
    Perform a chronological (time-series) split per market index to prevent
    look-ahead bias / data leakage.
    """
    print(f"Performing temporal train/test split (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%)...")
    train_indices = []
    test_indices = []

    for _, group in df.groupby("Index"):
        n = len(group)
        split_point = int(n * (1.0 - test_size))
        train_indices.extend(group.index[:split_point])
        test_indices.extend(group.index[split_point:])

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    print(f"Train set: {len(train_df):,} samples")
    print(f"Test set:  {len(test_df):,} samples")
    return train_df, test_df


def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Create a robust Scikit-Learn Pipeline combining StandardScaler,
    OneHotEncoder, and LinearRegression.
    """
    transformers = [("num", StandardScaler(), numeric_features)]

    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, current_close: np.ndarray) -> dict:
    """
    Compute comprehensive regression and trading direction metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (MAPE)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    # Directional Accuracy: Did model predict the correct sign of (Close_{t+1} - Close_t)?
    actual_direction = np.sign(y_true - current_close)
    pred_direction = np.sign(y_pred - current_close)
    directional_acc = float(np.mean(actual_direction == pred_direction) * 100)

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE_percent": round(mape, 4),
        "R2_Score": round(r2, 6),
        "Directional_Accuracy_percent": round(directional_acc, 2),
    }


def extract_feature_importance(pipeline: Pipeline, numeric_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Extract learned regression coefficients and map them to transformed feature names.
    """
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = list(numeric_features)
    if "cat" in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_names)

    coefficients = regressor.coef_
    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Abs_Coefficient": np.abs(coefficients),
    }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

    return df_coef


def generate_plots(test_df: pd.DataFrame, output_dir: str, df_coef: pd.DataFrame):
    """
    Create visualization figures for model evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating evaluation plots...")

    # 1. Actual vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    sample_data = test_df.sample(min(5000, len(test_df)), random_state=42)
    sns.scatterplot(
        data=sample_data,
        x="Target_Next_Close",
        y="Predicted_Next_Close",
        hue="Index",
        alpha=0.6,
        palette="tab10",
        ax=ax,
        s=20,
    )
    min_val = min(sample_data["Target_Next_Close"].min(), sample_data["Predicted_Next_Close"].min())
    max_val = max(sample_data["Target_Next_Close"].max(), sample_data["Predicted_Next_Close"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    ax.set_title("Actual vs. Predicted Next-Day Close Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Next-Day Close", fontsize=12)
    ax.set_ylabel("Predicted Next-Day Close", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Index")
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Residuals Distribution Plot
    residuals = test_df["Target_Next_Close"] - test_df["Predicted_Next_Close"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram & KDE of residuals (clipped for outliers visibility)
    q01, q99 = residuals.quantile(0.01), residuals.quantile(0.99)
    filtered_res = residuals[(residuals >= q01) & (residuals <= q99)]
    sns.histplot(filtered_res, kde=True, color="#2b5c8f", bins=50, ax=ax1)
    ax1.axvline(0, color="red", linestyle="--", lw=1.5)
    ax1.set_title("Residuals Distribution (1st-99th percentile)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Residual (Actual - Predicted)")

    # Residuals vs Predicted
    sample_sub = test_df.sample(min(3000, len(test_df)), random_state=42)
    sample_res = sample_sub["Target_Next_Close"] - sample_sub["Predicted_Next_Close"]
    ax2.scatter(sample_sub["Predicted_Next_Close"], sample_res, alpha=0.4, color="#e67e22", s=15)
    ax2.axhline(0, color="black", linestyle="--", lw=1.5)
    ax2.set_title("Residuals vs. Predicted Price", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predicted Next-Day Close")
    ax2.set_ylabel("Residual")

    plt.savefig(os.path.join(output_dir, "residuals_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Time Series Forecast (Recent 200 trading days for top 3 indices)
    indices_to_plot = [idx for idx in ["NYA", "IXIC", "N225"] if idx in test_df["Index"].unique()]
    if not indices_to_plot:
        indices_to_plot = list(test_df["Index"].unique()[:3])

    fig, axes = plt.subplots(len(indices_to_plot), 1, figsize=(14, 4 * len(indices_to_plot)), sharex=False)
    if len(indices_to_plot) == 1:
        axes = [axes]

    for ax, idx_name in zip(axes, indices_to_plot):
        sub = test_df[test_df["Index"] == idx_name].sort_values("Date").tail(200)
        ax.plot(sub["Date"], sub["Target_Next_Close"], label="Actual Next-Day Close", color="#1f77b4", lw=2)
        ax.plot(sub["Date"], sub["Predicted_Next_Close"], label="Predicted Next-Day Close", color="#ff7f0e", linestyle="--", lw=1.8)
        ax.set_title(f"Next-Day Close Forecast: Index {idx_name} (Recent 200 Days)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")

    plt.xlabel("Date", fontsize=11)
    plt.savefig(os.path.join(output_dir, "time_series_forecast.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Top Feature Coefficients
    fig, ax = plt.subplots(figsize=(9, 6))
    top_coef = df_coef.head(12)
    sns.barplot(data=top_coef, x="Coefficient", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_title("Top Linear Regression Feature Coefficients", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient Weight")
    plt.savefig(os.path.join(output_dir, "feature_coefficients.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved evaluation plots to '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser(description="Train Scikit-Learn Linear Regression for Stock Market Next-Day Close")
    parser.add_argument("--data", type=str, default="Market.csv", help="Path to Market.csv dataset")
    parser.add_argument("--index", type=str, default="all", help="Target index symbol (e.g., 'all', 'NYA', 'IXIC')")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set fraction (chronological)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save model & artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" STOCK MARKET NEXT-DAY CLOSE PRICE FORECASTING (SCIKIT-LEARN) ")
    print("=" * 70)

    # 1. Load and Preprocess Data
    raw_df = load_and_preprocess_data(args.data, target_index=args.index)

    # 2. Feature Engineering
    df_features = engineer_features(raw_df)

    # 3. Chronological Train/Test Split
    train_df, test_df = temporal_train_test_split(df_features, test_size=args.test_size)

    # Define feature lists
    numeric_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Close_Lag1",
        "Close_Lag2",
        "Close_Lag5",
        "SMA_5",
        "SMA_10",
        "Daily_Return",
        "Intraday_Range",
        "Log_Volume",
    ]
    categorical_features = ["Index"] if args.index.lower() == "all" else []

    all_features = numeric_features + categorical_features
    target_col = "Target_Next_Close"

    X_train = train_df[all_features]
    y_train = train_df[target_col]
    X_test = test_df[all_features]
    y_test = test_df[target_col]

    # 4. Build and Train Pipeline
    print("\nBuilding Scikit-Learn Pipeline and fitting model...")
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    print("Model training completed successfully!")

    # 5. Predict on Train & Test
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_df["Predicted_Next_Close"] = train_preds
    test_df["Predicted_Next_Close"] = test_preds

    # 6. Evaluate Overall Metrics
    train_metrics = calculate_metrics(y_train.values, train_preds, train_df["Close"].values)
    test_metrics = calculate_metrics(y_test.values, test_preds, test_df["Close"].values)

    print("\n" + "=" * 50)
    print(" OVERALL MODEL EVALUATION ")
    print("=" * 50)
    print(f"{'Metric':<30} | {'Train':<12} | {'Test':<12}")
    print("-" * 60)
    for k in test_metrics:
        print(f"{k:<30} | {train_metrics[k]:<12} | {test_metrics[k]:<12}")
    print("=" * 50)

    # 7. Per-Index Performance Breakdown
    per_index_results = []
    print("\nPER-INDEX TEST SET BREAKDOWN:")
    print("-" * 75)
    print(f"{'Index':<12} | {'Count':<8} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R2':<8}")
    print("-" * 75)
    for idx_name, group in test_df.groupby("Index"):
        idx_metrics = calculate_metrics(
            group[target_col].values,
            group["Predicted_Next_Close"].values,
            group["Close"].values,
        )
        per_index_results.append({
            "Index": idx_name,
            "Sample_Count": len(group),
            **idx_metrics,
        })
        print(
            f"{idx_name:<12} | {len(group):<8} | {idx_metrics['MAE']:<10.2f} | "
            f"{idx_metrics['RMSE']:<10.2f} | {idx_metrics['MAPE_percent']:<9.2f}% | {idx_metrics['R2_Score']:<8.4f}"
        )
    print("-" * 75)

    df_per_index = pd.DataFrame(per_index_results)
    df_per_index.to_csv(os.path.join(args.output_dir, "per_index_metrics.csv"), index=False)

    # 8. Feature Coefficients
    df_coef = extract_feature_importance(pipeline, numeric_features, categorical_features)
    df_coef.to_csv(os.path.join(args.output_dir, "feature_coefficients.csv"), index=False)
    print("\nTop 5 Learned Coefficients (Standardized Scale):")
    for _, row in df_coef.head(5).iterrows():
        print(f"  - {row['Feature']:<20}: {row['Coefficient']:>10.4f}")

    # 9. Save Artifacts
    model_path = os.path.join(args.output_dir, "linear_regression_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nTrained model pipeline saved to: {model_path}")

    # Save summary JSON
    summary_data = {
        "dataset": args.data,
        "target_variable": target_col,
        "index_scope": args.index,
        "features": all_features,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "intercept": float(pipeline.named_steps["regressor"].intercept_),
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # 10. Generate and Save Visualizations
    generate_plots(test_df, args.output_dir, df_coef)

    print("\nExecution complete! All outputs generated successfully.")


if __name__ == "__main__":
    main()
