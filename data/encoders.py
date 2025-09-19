import collections
import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool

import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer as NLTKTweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from config import MAX_LENGTH, NUM_WORKERS, USE_STEMMING, setup_logger

logger = setup_logger(__name__, "VIOLET")
nltk.download('wordnet')


class BaseTokenizer(ABC):
    SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    PAD, UNK, CLS, SEP = SPECIALS

    def __init__(self, tokenizer_name: str):
        self._vocab = None

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._vocab_file = os.path.join(
            project_root, "data", "toks", f"{tokenizer_name}_vocab.json"
        )
        self._model_file = os.path.join(project_root, "data", "toks", f"{tokenizer_name}.json")

        self._detokenizer = TreebankWordDetokenizer()

    def decode(self, input_ids: list[int]) -> str:
        vocab = self.fetch_vocab

        itos = vocab.get("itos", [])
        if not itos:
            logger.warning("No vocabulary found. Not decoding")
            return

        tokens = []
        for i in input_ids:
            if i < 0 or i >= len(itos):
                continue
            tok = itos[i]
            if tok in self.SPECIALS:
                continue
            tokens.append(tok)
        return self._detokenizer.detokenize(tokens)

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
    def fetch_vocab(self) -> dict | None:
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

    def _build_vocab_from_dict(self, vocab_dict: dict) -> dict:
        itos = [None] * len(vocab_dict)
        for token, idx in vocab_dict.items():
            itos[idx] = token

        vocab = {"stoi": vocab_dict, "itos": itos}
        self.save_vocab(vocab)
        logger.info(f"Successfully built vocab for {self.name.upper()} with {len(itos)} tokens")
        return vocab


class TweetTokenizer(BaseTokenizer):

    def __init__(self, preserve_case=False, reduce_len=True, strip_handles=True):
        self.name = "tweet"

        super().__init__(self.name)
        self.tokenizer = NLTKTweetTokenizer(
            preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles
        )
        self.use_stemming = USE_STEMMING
        if self.use_stemming:
            self.stemmer = SnowballStemmer("english")
            self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        if self.use_stemming:
            # Apply stemming first, then lemmatization
            tokens = [self.stemmer.stem(token) for token in tokens]
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def _tokenize_batch(self, texts_batch):
        batch_tokens = []
        for text in texts_batch:
            tokens = self.tokenize(text)
            batch_tokens.extend(tokens)
        return batch_tokens

    def build_vocab(self, texts: list[str], max_vocab: int, min_freq: int) -> dict:
        """
        Build vocabulary from texts

        Args:
            texts: List of text strings to build vocabulary from
            max_vocab: Maximum vocabulary size to keep
            min_freq: Minimum frequency threshold for tokens

        Returns:
            Dictionary containing 'stoi' (string to index) and 'itos' (index to string) mappings
        """
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
        return vocab

    def numericalize(self, tokens: list[str], stoi: dict) -> list[int]:
        return [stoi.get(token, stoi[self.UNK]) for token in tokens]

    def encode(self, text: str, vocab: dict, max_len: int = MAX_LENGTH) -> dict:
        stoi = vocab["stoi"]

        ids = [stoi[self.CLS]]
        tokens = self.tokenize(text)
        ids += self.numericalize(tokens, stoi)
        ids += [stoi[self.SEP]]

        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = stoi[self.SEP]

        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [stoi[self.PAD]] * pad_len
            attn += [0] * pad_len

        return {"input_ids": ids, "attention_mask": attn}


class BPETokenizer(BaseTokenizer):
    def __init__(self):
        self.name = "bpe"
        super().__init__(self.name)
        self.tokenizer = None

        if os.path.exists(self._model_file):
            self.tokenizer = Tokenizer.from_file(self._model_file)
            logger.info(f"Loaded BPE tokenizer from {self._model_file}")

    def _ensure_tokenizer_loaded(self):
        if self.tokenizer is None:
            if not os.path.exists(self._model_file):
                raise FileNotFoundError("Tokenizer file not found. Build the vocabulary first")
            self.tokenizer = Tokenizer.from_file(self._model_file)

    def tokenize(self, text: str) -> list[str]:
        if self.tokenizer is None:
            self._ensure_tokenizer_loaded()

        enc = self.tokenizer.encode(text)
        return enc.tokens

    def build_vocab(self, texts: list[str], max_vocab: int, min_frequency: int) -> dict:
        """
        Train BPE tokenizer and build vocabulary with byte-level processing.

        Args:
            texts: List of text strings to train tokenizer on
            max_vocab: Maximum vocabulary size for BPE training
            min_frequency: Minimum frequency threshold for subword merges

        Returns:
            Dictionary containing 'stoi' (string to index) and 'itos' (index to string) mappings
        """
        tok = Tokenizer(BPE(unk_token=self.UNK))
        tok.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=max_vocab,
            min_frequency=min_frequency,
            special_tokens=list(self.SPECIALS) + ["[MASK]"],
            show_progress=True,
        )
        tok.train_from_iterator(texts, trainer)

        tok.post_processor = TemplateProcessing(
            single=f"{self.CLS} $A {self.SEP}",
            pair=f"{self.CLS} $A {self.SEP} $B {self.SEP}",
            special_tokens=[
                (self.CLS, tok.token_to_id(self.CLS)),
                (self.SEP, tok.token_to_id(self.SEP)),
            ],
        )

        os.makedirs(os.path.dirname(self._model_file), exist_ok=True)
        tok.save(self._model_file)
        self.tokenizer = tok
        logger.info(f"Saved BPE tokenizer model to {self._model_file}")

        vocab_dict = tok.get_vocab()
        return self._build_vocab_from_dict(vocab_dict)

    def encode(self, text: str, vocab: dict, max_len: int = MAX_LENGTH) -> dict:
        self._ensure_tokenizer_loaded()

        pad_id = vocab["stoi"][self.PAD]
        enc = self.tokenizer.encode("" if text is None else text)
        input_ids = enc.ids
        attn = enc.attention_mask

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            input_ids[-1] = vocab["stoi"][self.SEP]
            attn = attn[:max_len]
        if len(input_ids) < max_len:
            pad_n = max_len - len(input_ids)
            input_ids += [pad_id] * pad_n
            attn += [0] * pad_n

        return {"input_ids": input_ids, "attention_mask": attn}

    def decode(self, input_ids: list[int]) -> str:
        self._ensure_tokenizer_loaded()

        specials = set(self.SPECIALS)
        ids = []
        for i in input_ids:
            tok = self.fetch_vocab["itos"][i] if self.fetch_vocab else None
            if tok is None or tok in specials:
                continue
            ids.append(i)
        try:
            return self.tokenizer.decode(ids)
        except Exception:
            toks = [self.fetch_vocab["itos"][i] for i in ids if self.fetch_vocab]
            return " ".join(toks)


class GPT2Tokenizer(BaseTokenizer):
    def __init__(self):
        self.name = "gpt2"
        super().__init__(self.name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.PAD not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"pad_token": self.PAD})

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def build_vocab(self, texts: list[str], max_vocab: int, min_frequency: int) -> dict:
        vocab_dict = self.tokenizer.get_vocab()
        return self._build_vocab_from_dict(vocab_dict)

    def encode(self, text: str, vocab: dict = None, max_len: int = MAX_LENGTH) -> dict:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def decode(self, input_ids: list[int]) -> str:
        special_token_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        }
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id:
            special_token_ids.add(self.tokenizer.bos_token_id)

        filtered_ids = [id for id in input_ids if id not in special_token_ids]
        return self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
