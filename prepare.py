import argparse
import os

import pandas as pd

from config import (
    DATA_PATH,
    MAX_VOCAB_SIZE,
    MIN_FREQ,
    SEED,
    TEST_FILE,
    TRAIN_FILE,
    set_seed,
    setup_logger,
)
from data.tokenizers import BPETokenizer, TweetTokenizer
from data.utils import TextCleaner, load_data

logger = setup_logger(__name__, "GREEN")


def clean_and_save_data(data_path: str, out_path: str) -> pd.DataFrame | None:

    if os.path.exists(out_path):
        logger.info(f"Using cached {out_path}")
        df = load_data(out_path, is_raw=False)
        return df

    df = load_data(data_path)
    if df is None:
        logger.error("Failed to load data files")
        return None

    logger.info("Cleaning text data!")
    cleaned_texts = TextCleaner.clean_texts_parallel(df['text'].tolist())
    df['cleaned_text'] = cleaned_texts
    df.drop(columns=["ids", "date", "flag", "user"], inplace=True)

    try:
        df.to_csv(out_path, index=False)
        logger.info(f"Cleaned data saved to {out_path}")
    except OSError as e:
        logger.error(f"Failed to save cleaned data to {out_path}: {e}")
        raise

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare sentiment analysis data")
    parser.add_argument(
        "--tokenizer",
        choices=["tweet", "bpe"],
        default="tweet",
        help="Tokenizer to use for vocabulary building",
    )

    args = parser.parse_args()

    set_seed(SEED)
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    train_out_path = os.path.join(DATA_PATH, "train.csv")
    test_out_path = os.path.join(DATA_PATH, "test.csv")

    train_df = clean_and_save_data(train_path, train_out_path)
    clean_and_save_data(test_path, test_out_path)

    if args.tokenizer == "tweet":
        tokenizer = TweetTokenizer()
    elif args.tokenizer == "bpe":
        tokenizer = BPETokenizer()
    else:
        logger.error("Invalid tokenizer found. Not proceeding with vocabulary creation")
        return

    if not tokenizer.is_present:
        logger.info(f"{args.tokenizer.upper()} selected. Proceeding with vocabulary creation")
        tokenizer.build_vocab(train_df['cleaned_text'].tolist(), MAX_VOCAB_SIZE, MIN_FREQ)

    logger.info("Data preparation completed successfully")


if __name__ == "__main__":
    main()
