import collections
import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool

from nltk.stem import SnowballStemmer
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

from config import MAX_LENGTH, NUM_WORKERS, USE_STEMMING, setup_logger

logger = setup_logger(__name__, "VIOLET")


class BaseTokenizer(ABC):
    SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
    PAD, UNK, CLS, SEP = SPECIALS

    def __init__(self, tokenizer_name: str):
        self._vocab = None
        self._vocab_file = os.path.join("data", "toks", f"{tokenizer_name}_vocab.json")
        self._model_file = os.path.join("data", "toks", f"{tokenizer_name}.json")

        self._detokenizer = TreebankWordDetokenizer()

    def decode(self, input_ids: list[int]) -> str:
        vocab = self.get_vocab

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
        super().__init__("bpe")
        self.tokenizer = None

        if os.path.exists(self._model_file):
            self.tokenizer = Tokenizer.from_file(self._model_file)
            logger.info(f"Loaded BPE tokenizer from {self._model_file}")

    def tokenize(self, text: str) -> list[str]:
        if self.tokenizer is None:
            raise ValueError("Invalid tokenizer")

        enc = self.tokenizer.encode(text)
        return enc.tokens

    def build_vocab(self, texts: list[str], max_vocab: int, min_frequency: int) -> dict:
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
        itos = [None] * len(vocab_dict)
        for token, idx in vocab_dict.items():
            itos[idx] = token
        vocab = {"stoi": vocab_dict, "itos": itos}
        self.save_vocab(vocab)
        logger.info(f"Successfully trained BPE tokenizer with {len(vocab_dict)} tokens")
        return vocab

    def encode(self, text: str, vocab: dict, max_len: int = MAX_LENGTH) -> dict:
        if self.tokenizer is None:
            if not os.path.exists(self._model_file):
                raise FileNotFoundError("Tokenizer file not found. Build the vocabulary first")
            self.tokenizer = Tokenizer.from_file(self._model_file)

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
        if self.tokenizer is None:
            if not os.path.exists(self._model_file):
                raise FileNotFoundError("Tokenizer file not found. Build the vocabulary first")
            self.tokenizer = Tokenizer.from_file(self._model_file)

        specials = set(self.SPECIALS)
        ids = []
        for i in input_ids:
            tok = self.get_vocab["itos"][i] if self.get_vocab else None
            if tok is None or tok in specials:
                continue
            ids.append(i)
        try:
            return self.tokenizer.decode(ids)
        except Exception:
            toks = [self.get_vocab["itos"][i] for i in ids if self.get_vocab]
            return " ".join(toks)
