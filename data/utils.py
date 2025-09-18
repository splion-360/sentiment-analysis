import os
import re
import string
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import BATCH_SIZE, NUM_WORKERS, SEED, TRAIN_SPLIT, set_seed, setup_logger
from data.tokenizers import BPETokenizer, TweetTokenizer

logger = setup_logger(__name__, "RED")


class TextCleaner:
    URL_PATTERN = (
        r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?'
    )
    MENTION_PATTERN = r"@\w+"
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    _URL_REGEX = re.compile(URL_PATTERN, re.IGNORECASE)
    _MENTION_REGEX = re.compile(MENTION_PATTERN)
    _EMAIL_REGEX = re.compile(EMAIL_PATTERN)
    _LEADING_NONALNUM_REGEX = re.compile(r'^[^A-Za-z0-9]+')
    _MULTIPLE_EXCLAMATION_REGEX = re.compile(r'!{2,}')
    _MULTIPLE_QUESTION_REGEX = re.compile(r'\?{2,}')
    _PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
    _WHITESPACE_REGEX = re.compile(r'\s+')
    _SINGLE_MARK_REGEX = re.compile(r"(^|[^!?])[!?](?![!?])")

    @staticmethod
    def contains_url(text: str) -> bool:
        return bool(TextCleaner._URL_REGEX.search(text))

    @staticmethod
    def contains_mention(text: str) -> bool:
        return bool(TextCleaner._MENTION_REGEX.search(text))

    @staticmethod
    def contains_email(text: str) -> bool:
        return bool(TextCleaner._EMAIL_REGEX.search(text))

    @staticmethod
    def contains_multiple_marks(text: str) -> bool:
        return bool(
            TextCleaner._MULTIPLE_EXCLAMATION_REGEX.search(text)
            or TextCleaner._MULTIPLE_QUESTION_REGEX.search(text)
        )

    @staticmethod
    def contains_single_mark(text: str) -> bool:
        return bool(TextCleaner._SINGLE_MARK_REGEX.search(text))

    @staticmethod
    def remove_urls(text: str) -> str:
        return TextCleaner._URL_REGEX.sub('', text)

    @staticmethod
    def remove_mentions(text: str) -> str:
        return TextCleaner._MENTION_REGEX.sub('', text)

    @staticmethod
    def remove_emails(text: str) -> str:
        return TextCleaner._EMAIL_REGEX.sub('', text)

    @staticmethod
    def remove_leading_nonalnum(text: str) -> str:
        return TextCleaner._LEADING_NONALNUM_REGEX.sub('', text)

    @staticmethod
    def remove_multiple_marks(text: str) -> str:
        text = TextCleaner._MULTIPLE_EXCLAMATION_REGEX.sub('!', text)
        text = TextCleaner._MULTIPLE_QUESTION_REGEX.sub('?', text)
        return text

    @staticmethod
    def remove_punctuations(text: str) -> str:
        return TextCleaner._PUNCTUATION_REGEX.sub('', text)

    @staticmethod
    def clean_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = TextCleaner._URL_REGEX.sub('', text)
        text = TextCleaner._MENTION_REGEX.sub('', text)
        text = TextCleaner._EMAIL_REGEX.sub('', text)
        text = TextCleaner._LEADING_NONALNUM_REGEX.sub('', text)
        text = TextCleaner._MULTIPLE_EXCLAMATION_REGEX.sub('!', text)
        text = TextCleaner._MULTIPLE_QUESTION_REGEX.sub('?', text)

        cleaned = TextCleaner._WHITESPACE_REGEX.sub(' ', text).strip()
        return cleaned

    @classmethod
    def clean_texts_parallel(cls, texts: list[str]) -> list[str]:
        if not texts:
            return []

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

    @staticmethod
    def _clean_batch(texts_batch: list[str]) -> list[str]:
        return [TextCleaner.clean_text(text) for text in texts_batch]

    @classmethod
    def calculate_tokens_parallel(
        cls, texts: list[str], tokenizer: TweetTokenizer | BPETokenizer
    ) -> list[int]:

        if not texts:
            return []

        batch_size = max(1, len(texts) // (NUM_WORKERS * 4))
        text_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        logger.info(
            f"Calculating token lengths for {len(texts)} texts using {NUM_WORKERS} workers..."
        )

        with Pool(processes=NUM_WORKERS) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(
                        cls._calculate_tokens_batch,
                        [(batch, tokenizer) for batch in text_batches],
                    ),
                    total=len(text_batches),
                    desc="Tokenizing batches",
                    unit="batches",
                )
            )
        token_lengths = []
        for batch_result in batch_results:
            token_lengths.extend(batch_result)

        return token_lengths

    @staticmethod
    def _calculate_tokens_batch(args) -> list[int]:
        texts_batch, tokenizer = args
        return [len(tokenizer.tokenize(text)) for text in texts_batch]


def load_data(
    data_dir: str, is_raw: bool = True, columns: list[str] = None, encoding: str = "ISO-8859-1"
) -> pd.DataFrame:

    if not os.path.exists(data_dir):
        logger.error(f"Data file not found: {data_dir}")
        raise FileNotFoundError(f"Data file not found: {data_dir}")

    if is_raw and not columns:
        columns = ["target", "ids", "date", "flag", "user", "text"]

    try:

        df = pd.read_csv(data_dir, names=columns, encoding=encoding)
        df.dropna(inplace=True)
        logger.info(f"Dataset loaded from: {data_dir} with {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Failed reading {data_dir} with {encoding}: {e}")
        raise


def split_data(
    df: pd.DataFrame, train_split: float = TRAIN_SPLIT, is_stratify: bool = True
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:

    texts = df['cleaned_text'].values.astype('str')
    targets = df['target'].values

    stratify = None

    if is_stratify:
        stratify = targets

    train_texts, val_texts, train_targets, val_targets = train_test_split(
        texts,
        targets,
        train_size=train_split,
        random_state=SEED,
        stratify=stratify,
        shuffle=True,
    )
    return (train_texts, train_targets), (val_texts, val_targets)


class SentimentDataset(Dataset):

    def __init__(
        self,
        texts: np.ndarray[str],
        targets: np.ndarray[int],
        tokenizer: TweetTokenizer | BPETokenizer | None,
        target_mapping: dict = None,
        max_length: int = None,
        quantile: float = 0.99,
    ):

        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab
        if self.vocab is None:
            raise ValueError("No vocabulary found for tokenizer. Ensure prepare.py is run!")

        self.target_mapping = target_mapping or {}
        self.max_length = max_length if max_length else self.get_max_length(quantile)

    def get_max_length(self, quantile) -> int:
        token_lengths = TextCleaner.calculate_tokens_parallel(self.texts.tolist(), self.tokenizer)
        return int(np.quantile(token_lengths, quantile))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        target = self.targets[idx]

        if self.target_mapping:
            target = self.target_mapping.get(target, target)

        encoding = self.tokenizer.encode(text, self.vocab, max_len=self.max_length)

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def get_dataloaders(
    data_path: str,
    tokenizer: TweetTokenizer | BPETokenizer,
    batch_size: int = BATCH_SIZE,
    target_mapping: dict = None,
    train_split: float = TRAIN_SPLIT,
    stratify: bool = True,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    is_train: bool = True,
    max_length: int = None,
):
    set_seed(SEED)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Invalid data path {data_path}")

    data = load_data(data_path, is_raw=False)

    if not is_train:
        texts = data['cleaned_text'].values
        targets = data['target'].values

        if target_mapping:
            valid_indices = []
            for i, target in enumerate(targets):
                if target in target_mapping:
                    valid_indices.append(i)

            texts = texts[valid_indices]
            targets = targets[valid_indices]
            logger.info(
                f"Filtered test data from {len(data)} to {len(texts)}"
                " samples (removed unseen labels)",
            )

        dataset = SentimentDataset(
            texts, targets, tokenizer, target_mapping=target_mapping, max_length=max_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(SEED),
        )

        logger.info(f"Created dataloader with {len(dataset)} samples")
        logger.info(f"Max length set to: {dataset.max_length}")
        return dataloader

    else:
        (train_texts, train_targets), (val_texts, val_targets) = split_data(
            data, train_split=train_split, is_stratify=stratify
        )

        train_dataset = SentimentDataset(
            train_texts,
            train_targets,
            tokenizer,
            target_mapping=target_mapping,
            max_length=max_length,
        )
        val_dataset = SentimentDataset(
            val_texts,
            val_targets,
            tokenizer,
            target_mapping=target_mapping,
            max_length=train_dataset.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(SEED),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(SEED),
        )

        assert train_dataset.max_length == val_dataset.max_length, "Mismatched sequence lengths"

        logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        logger.info(f"Max length set to: {train_dataset.max_length}")

        return train_loader, val_loader
