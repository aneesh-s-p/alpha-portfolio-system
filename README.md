# Alpha + Portfolio System

A cross-sectional machine-learning stock/ETF selection pipeline. Every month it retrains an ensemble on a rolling window, scores a 29-asset universe by the probability of a positive forward return, then evaluates the signal with statistical validation, a long-only top-N portfolio simulation, and a permutation test.

> **Disclaimer:** research and educational project. Nothing here is investment advice, and backtested results do not guarantee future performance.

## How it works

1. **Data** – daily adjusted prices from Yahoo Finance (`yfinance`) per ticker, plus the S&P 500 (`^GSPC`) as the market benchmark.
2. **Features** (all scale-free, so they are comparable across assets):
   price relative to its 20/50/200-day SMA, RSI, normalised MACD, 20-day return volatility, relative volume, 20-day relative strength vs. the S&P 500, 3/6/12-month momentum and rolling z-scores, plus trend/volatility regime flags.
3. **Regime detection** – trend (close vs. 200-day SMA) × volatility (vs. an *expanding* median, so no future data is used) gives four market regimes.
4. **Target** – whether the forward return over `HORIZON` days is positive.
5. **Walk-forward training** – every `REBALANCE_STEP` trading days a pooled model is retrained on a `WINDOW`-day rolling window. The final `HORIZON` days are held out of training so labels never reach into the prediction date.
6. **Ensemble** – equal-weighted average of Logistic Regression, Random Forest and Gradient Boosting probabilities.
7. **Validation** – AUC, pooled correlation, per-date Spearman rank IC (with t-stat), monthly top-vs-bottom spread, and decile analysis.
8. **Portfolio simulation** – long the top-N assets by probability, capped weights, turnover-based transaction costs, and a causal volatility-target overlay (trailing realised vol, no leverage). Reports Sharpe (net of a risk-free rate), max drawdown, turnover, and an equal-weight universe benchmark.
9. **Permutation test** – within each date, forward returns are shuffled among the assets. This tests whether the model's *selection* beats random picks while keeping the market's return on each date intact.

## Project structure

```
alpha-portfolio-system/
├── main.py                    # entry point
├── config.py                  # all tunable parameters
├── engine/
│   ├── alpha_engine.py        # rolling retraining + alpha dataset
│   ├── validation_engine.py   # AUC, IC, spread, deciles
│   └── portfolio_engine.py    # top-N simulation, vol target, permutation test
├── src/
│   ├── data_loader.py         # yfinance download
│   ├── feature_engineering.py # indicators, momentum, forward return
│   ├── regime_detection.py    # trend x volatility regimes
│   ├── model.py               # model training
│   └── backtester.py          # standalone regime-threshold backtest (not used by main.py)
├── tests/
│   └── test_pipeline.py       # offline tests on synthetic data
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/<your-username>/alpha-portfolio-system.git
cd alpha-portfolio-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

A full run downloads data for all tickers and retrains three models at every rebalance date, so it takes a while (the permutation test is the slowest part; lower `N_PERMUTATIONS` in `config.py` for a quick run).

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use synthetic prices (no internet needed) and check that volatility targeting and regime detection use no future information, leverage is capped, costs scale with turnover, raw price levels are kept out of the model, and the pipeline runs end to end.

## Configuration

Edit `config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `START_DATE` / `END_DATE` | 2010-01-01 / 2024-12-31 | Data range |
| `WINDOW` | 756 | Training window (trading days, ~3 years) |
| `REBALANCE_STEP` | 21 | Trading days between retrains/rebalances (~monthly) |
| `HORIZON` | = `REBALANCE_STEP` | Forward-return horizon; must be ≤ `REBALANCE_STEP` so periods don't overlap |
| `TOP_N` | 5 | Number of assets held |
| `MAX_WEIGHT` | 0.35 | Per-asset weight cap |
| `TRANSACTION_COST` | 0.001 | Cost per unit of traded notional |
| `RISK_FREE_RATE` | 0.02 | Annual rate used in the Sharpe ratio (approximate) |
| `TARGET_VOL` | 0.15 | Annual volatility target |
| `MAX_LEVERAGE` | 1.0 | Vol targeting can reduce exposure, never lever up |
| `VOL_LOOKBACK` / `VOL_MIN_PERIODS` | 12 / 6 | Trailing months for realised vol / warm-up |
| `N_PERMUTATIONS` | 500 | Permutation-test shuffles |
| `REGIME_THRESHOLDS` | see file | Probability thresholds per regime (used by `src/backtester.py`) |
| `RANDOM_SEED` | 42 | Reproducibility |
| `TICKERS` | 29 assets | Stocks, equity/sector/international ETFs, bonds, gold, silver, REITs |

## Output

`main.py` logs the validation report, then simulated final capital, total return, Sharpe, max drawdown, turnover, an equal-weight benchmark, and the permutation-test p-value. Results depend on the download date and library versions.

<!-- After your first real run, paste the key results here (AUC, mean rank IC, Sharpe vs benchmark, p-value). -->

## Limitations

- **Survivorship / selection bias:** the universe was chosen with hindsight (several current large-cap winners), which flatters results.
- **Simplified execution:** costs are a flat rate on turnover; there is no slippage or market impact, and weight drift between rebalances is ignored.
- **Risk-free rate** is a constant approximation.
- **No hyper-parameter search or separate hold-out period:** the settings in `config.py` were not tuned, but results are one historical path rather than independent confirmation.
- Data comes from Yahoo Finance via `yfinance`, which can change or throttle.

## License

MIT – see [LICENSE](LICENSE).
