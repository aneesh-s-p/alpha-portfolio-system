import logging

from engine.alpha_engine import generate_alpha_dataset
from engine.validation_engine import evaluate_alpha
from engine.portfolio_engine import simulate_long_top_n, permutation_test
from config import TICKERS, TOP_N, N_PERMUTATIONS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("===== GENERATING ALPHA DATASET =====")
    alpha_df = generate_alpha_dataset(TICKERS)

    if alpha_df is None or len(alpha_df) == 0:
        logger.error("Alpha dataset is empty. Exiting.")
        return

    logger.info("===== VALIDATION =====")
    evaluate_alpha(alpha_df)

    logger.info("===== PORTFOLIO SIMULATION =====")
    simulate_long_top_n(alpha_df, top_n=TOP_N)

    logger.info("===== PERMUTATION TEST =====")
    permutation_test(alpha_df, top_n=TOP_N, n_tests=N_PERMUTATIONS)


if __name__ == "__main__":
    main()
