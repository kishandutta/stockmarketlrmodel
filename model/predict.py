"""
Inference Script for Next-Day Stock Market Close Price Predictor

This script loads the trained Scikit-Learn Linear Regression model pipeline
and produces forecasts on either:
1. The most recent available trading date in Market.csv for any index.
2. A custom CSV file or specific feature values.
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np


def load_model(model_path: str = "outputs/linear_regression_model.joblib"):
    """Load the serialized scikit-learn pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please run 'python train_linear_regression.py' first."
        )
    return joblib.load(model_path)


def predict_latest_for_index(csv_path: str, index_symbol: str, model_path: str):
    """
    Extract the latest data from Market.csv for an index, compute features,
    and predict the upcoming day's Close price.
    """
    model = load_model(model_path)
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Index"] == index_symbol].sort_values("Date").dropna(subset=["Open", "High", "Low", "Close"]).copy()

    if len(df) < 10:
        raise ValueError(f"Not enough historical data for index '{index_symbol}'. Found {len(df)} rows.")

    # Calculate required features for the last row
    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag5"] = df["Close"].shift(5)
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]
    df["Log_Volume"] = np.log1p(df["Volume"].fillna(0).clip(lower=0))

    latest_row = df.iloc[-1:]

    feature_cols = [
        "Open", "High", "Low", "Close",
        "Close_Lag1", "Close_Lag2", "Close_Lag5",
        "SMA_5", "SMA_10", "Daily_Return", "Intraday_Range", "Log_Volume",
        "Index"
    ]

    predicted_next_close = float(model.predict(latest_row[feature_cols])[0])
    current_close = float(latest_row["Close"].values[0])
    diff = predicted_next_close - current_close
    pct_change = (diff / current_close) * 100
    latest_date = latest_row["Date"].values[0]

    print("\n" + "=" * 50)
    print(f" FORECAST FOR INDEX: {index_symbol} ")
    print("=" * 50)
    print(f"Latest Available Date : {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}")
    print(f"Latest Actual Close   : {current_close:,.2f}")
    print(f"Predicted Next Close  : {predicted_next_close:,.2f}")
    print(f"Expected Price Change : {diff:+,.2f} ({pct_change:+.2f}%)")
    print(f"Expected Direction    : {'BULLISH (UP)' if diff > 0 else 'BEARISH (DOWN)'}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference using the trained Linear Regression model")
    parser.add_argument("--data", type=str, default="Market.csv", help="Path to Market.csv")
    parser.add_argument("--model", type=str, default="outputs/linear_regression_model.joblib", help="Path to trained model")
    parser.add_argument("--index", type=str, default="NYA", help="Stock index symbol (e.g. NYA, IXIC, N225)")
    args = parser.parse_args()

    predict_latest_for_index(args.data, args.index, args.model)


if __name__ == "__main__":
    main()
