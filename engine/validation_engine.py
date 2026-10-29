import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_alpha(alpha_df):

    print("\n===== Alpha Validation Report =====")

    if alpha_df is None or len(alpha_df) == 0:
        print("Alpha dataset is empty.")
        return

    alpha_df = alpha_df.copy()

    # Drop missing values safely
    alpha_df = alpha_df.dropna(
        subset=["probability", "forward_return", "date"]
    )

    if len(alpha_df) < 10:
        print("Not enough samples for validation.")
        return

    print(f"Total samples: {len(alpha_df)}")

    # ---------------------------------------------------
    # Binary Target for AUC
    # ---------------------------------------------------

    alpha_df["target"] = (alpha_df["forward_return"] > 0).astype(int)

    # ---------------------------------------------------
    # AUC
    # ---------------------------------------------------

    try:
        if alpha_df["target"].nunique() > 1:
            auc = roc_auc_score(
                alpha_df["target"],
                alpha_df["probability"]
            )
            print("AUC:", round(auc, 4))
        else:
            print("AUC: Not computable (only one class present)")
    except Exception as e:
        print("AUC: Not computable")

    # ---------------------------------------------------
    # Pooled Correlation
    # ---------------------------------------------------

    correlation = alpha_df["probability"].corr(
        alpha_df["forward_return"]
    )

    if pd.isna(correlation):
        print("Pooled Correlation(probability, forward_return): Not computable")
    else:
        print("Correlation(probability, forward_return):",
              round(correlation, 4))

    # ---------------------------------------------------
    # Cross-Sectional Rank IC (Spearman, per date)
    # ---------------------------------------------------

    ics = []

    for date, group in alpha_df.groupby("date"):
        if len(group) < 5:
            continue
        if group["probability"].nunique() < 2 or \
           group["forward_return"].nunique() < 2:
            continue
        ic = group["probability"].corr(
            group["forward_return"], method="spearman"
        )
        if pd.notna(ic):
            ics.append(ic)

    print("\nCross-Sectional Rank IC:")

    if len(ics) > 1:
        ics = np.array(ics)
        mean_ic = ics.mean()
        ic_std = ics.std(ddof=1)
        t_stat = mean_ic / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else float("nan")
        print("Mean Rank IC:", round(mean_ic, 4))
        print("IC Std:", round(ic_std, 4))
        print("IC t-stat:", round(t_stat, 2))
        print("% months IC > 0:", round(np.mean(ics > 0) * 100, 2), "%")
    else:
        print("Not enough dates with enough assets to compute rank IC.")

    # ---------------------------------------------------
    # Cross-Sectional Monthly Spread
    # ---------------------------------------------------

    print("\nCross-Sectional Monthly Spread Test:")

    date_spreads = []

    for date, group in alpha_df.groupby("date"):

        if len(group) < 2:
            continue

        ranked = group.sort_values("probability")

        bottom = ranked.iloc[0]["forward_return"]
        top = ranked.iloc[-1]["forward_return"]

        spread = top - bottom
        date_spreads.append(spread)

    if len(date_spreads) > 0:

        date_spreads = np.array(date_spreads)

        avg_spread = date_spreads.mean()
        win_rate = np.mean(date_spreads > 0) * 100

        print("Average Top-Bottom Spread per Month:",
              round(avg_spread, 6))

        print("Win Rate (% months top beats bottom):",
              round(win_rate, 2), "%")

    else:
        print("Not enough data for cross-sectional test.")

    # ---------------------------------------------------
    # Decile Analysis
    # ---------------------------------------------------

    try:
        alpha_df["decile"] = pd.qcut(
            alpha_df["probability"],
            10,
            labels=False,
            duplicates="drop"
        )

        decile_returns = (
            alpha_df
            .groupby("decile")["forward_return"]
            .mean()
            .sort_index()
        )

        print("\nAverage Forward Return by Probability Decile:")
        print(decile_returns)

        if 0 in decile_returns.index and \
           max(decile_returns.index) in decile_returns.index:

            spread = (
                decile_returns.loc[max(decile_returns.index)]
                - decile_returns.loc[0]
            )

            print("\nTop Decile - Bottom Decile Spread:",
                  round(spread, 6))

    except Exception:
        print("Decile analysis not computable.")

    print("\nValidation complete.\n")
