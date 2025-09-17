import collections
import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool

from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer as NLTKTweetTokenizer
from tqdm import tqdm

from config import MAX_LENGTH, NUM_WORKERS, USE_STEMMING, setup_logger

logger = setup_logger(__name__, "VIOLET")


class BaseTokenizer(ABC):

    def __init__(self, tokenizer_name: str):
        self._vocab = None
        self._vocab_file = os.path.join(
            os.path.join("data", "toks"), f"{tokenizer_name}_vocab.json"
        )

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def build_vocab(self, texts: list[str], **kwargs) -> dict:
        pass

    @abstractmethod
    def encode(self, text: str, vocab: dict, **kwargs) -> dict:
        pass

    @property
    def is_present(self) -> bool:
        return os.path.exists(self._vocab_file)

    @property
    def get_vocab(self) -> dict | None:
        if self._vocab is not None:
            return self._vocab

        if os.path.exists(self._vocab_file):
            try:
                logger.info(f"Loading vocabulary from {self._vocab_file}")
                with open(self._vocab_file, encoding="utf-8") as f:
                    self._vocab = json.load(f)
            except OSError as e:
                logger.error(f"Failed to read vocabulary file {self._vocab_file}: {e}")
                raise
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid JSON format in vocabulary file {self._vocab_file}: {e}")
                raise

        return self._vocab

    def save_vocab(self, vocab: dict):
        try:
            os.makedirs(os.path.dirname(self._vocab_file), exist_ok=True)
            with open(self._vocab_file, "w", encoding="utf-8") as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            self._vocab = vocab
            logger.info(f"Saved the vocabulary successfully at {self._vocab_file}")
        except OSError as e:
            logger.error(f"Failed to save vocabulary to {self._vocab_file}: {e}")
            raise


class TweetTokenizer(BaseTokenizer):
    SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    PAD, UNK, CLS, SEP = SPECIALS

    def __init__(self, preserve_case=False, reduce_len=True, strip_handles=True):
        self.name = "tweet"

        super().__init__(self.name)
        self.tokenizer = NLTKTweetTokenizer(
            preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles
        )
        self.use_stemming = USE_STEMMING
        if self.use_stemming:
            self.stemmer = SnowballStemmer("english")

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def _tokenize_batch(self, texts_batch):
        batch_tokens = []
        for text in texts_batch:
            tokens = self.tokenize(text)
            batch_tokens.extend(tokens)
        return batch_tokens

    def build_vocab(self, texts: list[str], max_vocab: int, min_freq: int) -> None:
        counter = collections.Counter()

        logger.info(f"Building vocabulary from {len(texts)} texts using {NUM_WORKERS} workers...")

        batch_size = max(1, len(texts) // (NUM_WORKERS * 4))
        text_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        with Pool(processes=NUM_WORKERS) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(self._tokenize_batch, text_batches),
                    total=len(text_batches),
                    desc="Processing batches",
                    unit="batches",
                )
            )

        for batch_tokens in batch_results:
            counter.update(batch_tokens)

        logger.info(f"Found {len(counter)} unique tokens")
        itos = self.SPECIALS + [
            w for w, c in counter.most_common() if c >= min_freq and w not in self.SPECIALS
        ]
        if len(itos) > max_vocab:
            itos = itos[:max_vocab]

        stoi = {w: i for i, w in enumerate(itos)}
        vocab = {"stoi": stoi, "itos": itos}

        self.save_vocab(vocab)
        logger.info(f"Successfully built vocab for {self.name.upper()} with {len(itos)} tokens")

    def numericalize(self, tokens: list[str], stoi: dict) -> list[int]:
        return [stoi.get(token, stoi[self.unk]) for token in tokens]

    @classmethod
    def encode(cls, text: str, vocab: dict, max_len: int = MAX_LENGTH) -> dict:
        stoi = vocab["stoi"]

        ids = [stoi[cls.CLS]]
        tokens = cls.tokenize(text)
        ids += cls.numericalize(tokens, stoi)
        ids += [stoi[cls.SEP]]

        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = stoi[cls.SEP]

        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [stoi[cls.PAD]] * pad_len
            attn += [0] * pad_len

        return {"input_ids": ids, "attention_mask": attn}


class BPETokenizer(BaseTokenizer):

    def tokenize(self, text: str) -> list[str]:
        pass

    def build_vocab(self, texts: list[str], **kwargs) -> dict:
        pass

    def encode(self, text: str, vocab: dict, **kwargs) -> dict:
        pass
