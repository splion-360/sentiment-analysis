import argparse
import os

from config import DATA_PATH, MAX_VOCAB_SIZE, MIN_FREQ, TEST_FILE, TRAIN_FILE, setup_logger
from data.tokenizers import BPETokenizer, TweetTokenizer
from data.utils import clean_and_save_data

logger = setup_logger(__name__, "GREEN")


def main():
    parser = argparse.ArgumentParser(description="Prepare sentiment analysis data")
    parser.add_argument(
        "--tokenizer",
        choices=["tweet", "bpe"],
        default="tweet",
        help="Tokenizer to use for vocabulary building",
    )

    args = parser.parse_args()

    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    train_out_path = os.path.join(DATA_PATH, "train.csv")
    test_out_path = os.path.join(DATA_PATH, "test.csv")

    train_df = clean_and_save_data(train_path, train_out_path)
    _ = clean_and_save_data(test_path, test_out_path)

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
