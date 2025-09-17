import os
import re
import string
from multiprocessing import Pool

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from config import (
    NUM_WORKERS,
    TRAIN_SPLIT,
    setup_logger,
)
from data.tokenizers import BPETokenizer, TweetTokenizer

logger = setup_logger(__name__)


def load_data(
    data_dir: str, columns: list[str] = None, encoding: str = "ISO-8859-1"
) -> pd.DataFrame | None:
    if not os.path.exists(data_dir):
        logger.error(f"Data file not found: {data_dir}")
        raise FileNotFoundError(f"Data file not found: {data_dir}")

    if not columns:
        columns = ["target", "ids", "date", "flag", "user", "text"]

    try:
        df = pd.read_csv(data_dir, names=columns, encoding=encoding, header=None)
        logger.info(f"Dataset loaded from: {data_dir} with {len(df)} rows")
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Failed to parse CSV file {data_dir}: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to read file {data_dir}: {e}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {data_dir} with {encoding}: {e}")
        raise


def clean_and_save_data(data_path: str, out_path: str) -> pd.DataFrame | None:
    df = load_data(data_path)

    if df is None:
        logger.error("Failed to load data files")
        return None

    logger.info("Cleaning text data!")
    cleaned_texts = TextCleaner.clean_texts_parallel(df['text'].tolist())
    df['cleaned_text'] = cleaned_texts

    df.drop(columns=["ids", "date", "flag", "user"], errors='ignore', inplace=True)

    try:
        df.to_csv(out_path, index=False)
        logger.info(f"Cleaned data saved to {out_path}")
    except OSError as e:
        logger.error(f"Failed to save cleaned data to {out_path}: {e}")
        raise

    return df


def construct_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} with {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Failed to parse CSV file {data_path}: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to read file {data_path}: {e}")
        raise


def split_data(df: pd.DataFrame, train_split=TRAIN_SPLIT, stratify=True):
    if stratify and 'target' in df.columns:
        train_df, val_df = train_test_split(
            df, train_size=train_split, random_state=42, stratify=df['target']
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        logger.info(f"Stratified split - Train size: {len(train_df)}, Val size: {len(val_df)}")
        logger.info(f"Train class distribution: {train_df['target'].value_counts().to_dict()}")
        logger.info(f"Val class distribution: {val_df['target'].value_counts().to_dict()}")
    else:
        shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(len(shuffled_df) * train_split)
        train_df = shuffled_df[:train_size].copy()
        val_df = shuffled_df[train_size:].copy()

        logger.info(f"Random split - Train size: {len(train_df)}, Val size: {len(val_df)}")

    return train_df, val_df


class TextCleaner:

    URL_PATTERN = (
        r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?'
    )
    MENTION_PATTERN = r"@\w+"
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    EMOTICON_PATTERN = (
        r"(?:[<>]?[:;=8xX][\-o*']?[\)\]\(\[dDpP/\:\}\{@\|\\])"
        r"|(?:[\)\]\(\[dDpP/\:\}\{@\|\\][\-o*']?[:;=8xX][<>]?)"
        r"|(?:[:;=8xX][\-o*']?[\(/{\[])"
        r"|(?:[:;=8xX][\-o*]?['’][\(\[])"
    )

    @classmethod
    def contains_url(cls, text: str):
        return bool(re.search(cls.URL_PATTERN, text))

    @classmethod
    def contains_mention(cls, text: str):
        return bool(re.search(cls.MENTION_PATTERN, text))

    @classmethod
    def contains_email(cls, text: str):
        return bool(re.search(cls.EMAIL_PATTERN, text))

    @classmethod
    def contains_multiple_mark(cls, text: str):
        exclamation_pattern = r"!{2,}"
        qmark_pattern = r"\?{2,}"
        return bool(re.search(exclamation_pattern, text) or re.search(qmark_pattern, text))

    @classmethod
    def contains_single_mark(cls, text: str) -> bool:
        return bool(re.search(r"(^|[^!?])[!?](?![!?])", text))

    @classmethod
    def remove_urls(cls, text: str):
        return re.sub(cls.URL_PATTERN, '', text)

    @classmethod
    def remove_mentions(cls, text: str):
        return re.sub(cls.MENTION_PATTERN, '', text)

    @classmethod
    def remove_emails(cls, text: str):
        return re.sub(cls.EMAIL_PATTERN, '', text)

    @classmethod
    def remove_leading_nonalnum(cls, text: str):
        return re.sub(r'^[^A-Za-z0-9]+', '', text)

    @classmethod
    def remove_multiple_marks(cls, text: str):
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        return text

    @classmethod
    def remove_punctuations(cls, text: str):
        return re.sub(f"[{re.escape(string.punctuation)}]", '', text)

    @classmethod
    def clean_text(cls, text: str):
        text = cls.remove_urls(text)
        text = cls.remove_mentions(text)
        text = cls.remove_emails(text)
        text = cls.remove_leading_nonalnum(text)
        text = cls.remove_multiple_marks(text)
        cleaned = ' '.join(text.split())
        return cleaned

    @classmethod
    def clean_texts_parallel(cls, texts: list[str]) -> list[str]:
        batch_size = max(1, len(texts) // (NUM_WORKERS * 4))
        text_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        logger.info(f"Cleaning {len(texts)} texts using {NUM_WORKERS} workers...")

        with Pool(processes=NUM_WORKERS) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(cls._clean_batch, text_batches),
                    total=len(text_batches),
                    desc="Cleaning batches",
                    unit="batches",
                )
            )

        cleaned_texts = []
        for batch_result in batch_results:
            cleaned_texts.extend(batch_result)

        return cleaned_texts

    @classmethod
    def _clean_batch(cls, texts_batch: list[str]) -> list[str]:
        return [cls.clean_text(text) for text in texts_batch]


class SentimentDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BPETokenizer | TweetTokenizer | None,
        vocab: dict,
        max_length: int = 128,
        target_mapping: dict = None,
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.target_mapping = target_mapping or {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = row['cleaned_text']
        target = row['target']

        if self.target_mapping:
            target = self.target_mapping.get(target, target)

        encoding = self.tokenizer.encode(text, self.vocab, max_len=self.max_length)

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.long),
        }
