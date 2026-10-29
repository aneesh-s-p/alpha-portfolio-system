from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

from config import RANDOM_SEED

# Raw price/volume levels are not comparable across assets or over time
# (a $500 stock vs a $30 ETF), so they are excluded from the model inputs.
# Scale-free equivalents (price_to_sma_*, macd_norm, volume_ratio_20) are used.
NON_FEATURE_COLUMNS = [
    "forward_return", "target",
    "Open", "High", "Low", "Close", "Volume", "market_close",
    "sma_20", "sma_50", "sma_200", "macd",
]


def train_models(df):
    """
    Trains ensemble classification models on full rolling window.
    Returns trained models, fitted scaler, and feature column names.
    """

    df = df.copy()

    # Binary target
    df["target"] = (df["forward_return"] > 0).astype(int)

    # Keep numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    X = numeric_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    y = numeric_df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    logreg = LogisticRegression(max_iter=2000)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_SEED
    )

    # Train on full rolling window
    logreg.fit(X_scaled, y)
    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)

    return logreg, rf, gb, scaler, X.columns
