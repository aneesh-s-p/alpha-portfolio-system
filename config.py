"""
Global configuration file for Alpha + Portfolio system.
All tunable parameters must live here.
"""

# ============================================
# Data
# ============================================

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


# ============================================
# Rolling Window & Training
# ============================================

WINDOW = 756              # 3 years of trading days
REBALANCE_STEP = 21       # ~Monthly (21 trading days)
HORIZON = REBALANCE_STEP  # Forward return horizon. Must be <= REBALANCE_STEP,
                          # otherwise consecutive periods overlap and compounded
                          # returns / Sharpe are overstated.


# ============================================
# Transaction & Capital
# ============================================

TRANSACTION_COST = 0.001  # Cost per unit of traded notional (0.1%)
INITIAL_CAPITAL = 100000
TOP_N = 5
RISK_FREE_RATE = 0.02     # Annual, approximate; used in the Sharpe ratio


# ============================================
# Risk Controls
# ============================================

TARGET_VOL = 0.15         # 15% annual volatility target
MAX_WEIGHT = 0.35         # Per-asset weight cap
MAX_LEVERAGE = 1.0        # Vol targeting may de-lever but never lever up
VOL_LOOKBACK = 12         # Trailing months used to estimate realized vol
VOL_MIN_PERIODS = 6       # Months of history needed before scaling starts


# ============================================
# Regime Detection
# ============================================

REGIME_VOL_MIN_PERIODS = 60   # Min history for the expanding volatility median

# Probability thresholds per regime (used by src/backtester.py)
REGIME_THRESHOLDS = {
    "1_1": 0.55,
    "1_-1": 0.65,
    "-1_1": 0.70,
    "-1_-1": 0.75
}


# ============================================
# Validation
# ============================================

N_PERMUTATIONS = 500


# ============================================
# Reproducibility
# ============================================

RANDOM_SEED = 42


# ============================================
# Asset Universe
# ============================================

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLY", "XLI", "XLV", "XLP", "XLU",
    "EFA", "EEM", "VEA", "VWO",
    "TLT", "IEF", "LQD", "HYG",
    "GLD", "SLV", "VNQ"
]
