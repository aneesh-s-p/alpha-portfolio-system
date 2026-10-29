import pandas as pd
import numpy as np

from src.data_loader import download_stock_data
from src.feature_engineering import (
    add_technical_indicators,
    add_relative_strength,
    add_forward_return,
    add_momentum_features
)
from src.regime_detection import detect_market_regime
from src.model import train_models

from config import WINDOW, HORIZON, REBALANCE_STEP


def generate_alpha_dataset(tickers):

    print("\n===== Generating Cross-Sectional Alpha Dataset =====")

    if HORIZON > REBALANCE_STEP:
        raise ValueError(
            f"HORIZON ({HORIZON}) must be <= REBALANCE_STEP ({REBALANCE_STEP}); "
            "otherwise consecutive forward returns overlap."
        )

    # -----------------------------
    # Load and preprocess all tickers
    # -----------------------------
    all_data = []

    for ticker in tickers:
        df = download_stock_data(ticker)

        if df is None or df.empty:
            print(f"Skipping {ticker}: no data downloaded")
            continue

        df = add_technical_indicators(df)
        df = add_relative_strength(df)
        df = add_momentum_features(df)
        df = add_forward_return(df, horizon=HORIZON)
        df = detect_market_regime(df)

        df["ticker"] = ticker
        df.dropna(inplace=True)

        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data)
    full_df.sort_index(inplace=True)

    results = []

    unique_dates = sorted(full_df.index.unique())

    # -----------------------------
    # Monthly cross-sectional retraining
    # -----------------------------
    for i in range(WINDOW + HORIZON, len(unique_dates) - HORIZON, REBALANCE_STEP):

        current_date = unique_dates[i]

        train_start_date = unique_dates[i - WINDOW - HORIZON]
        train_end_date = unique_dates[i - HORIZON]

        train_data = full_df[
            (full_df.index >= train_start_date) &
            (full_df.index < train_end_date)
        ].copy()

        if len(train_data) < 200:
            continue

        # Train pooled model
        logreg, rf, gb, scaler, feature_cols = train_models(train_data)

        # Current cross-section (all assets at current_date)
        current_data = full_df[full_df.index == current_date].copy()

        if len(current_data) == 0:
            continue

        X_current = current_data.select_dtypes(include=[np.number])
        X_current = X_current.reindex(columns=feature_cols, fill_value=0)

        X_scaled = scaler.transform(X_current)

        log_prob = logreg.predict_proba(X_scaled)[:, 1]
        rf_prob = rf.predict_proba(X_scaled)[:, 1]
        gb_prob = gb.predict_proba(X_scaled)[:, 1]

        ensemble_prob = (log_prob + rf_prob + gb_prob) / 3

        for j in range(len(current_data)):
            results.append({
                "date": current_date,
                "ticker": current_data.iloc[j]["ticker"],
                "probability": float(ensemble_prob[j]),
                "forward_return": float(current_data.iloc[j]["forward_return"])
        })


    alpha_df = pd.DataFrame(results)

    print("Cross-sectional alpha dataset generated.")
    print("Total samples:", len(alpha_df))

    return alpha_df
