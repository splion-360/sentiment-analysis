import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import OrderedDict

from config import (
    DATA_PATH,
    DEVICE,
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_PATH,
    SEED,
    TOKENIZER,
    set_seed,
    setup_logger,
)
from data.tokenizers import BPETokenizer, TweetTokenizer
from data.utils import TextCleaner, get_dataloaders
from training.models.transformer import Transformer

logger = setup_logger(__name__, "TURQUOISE")

_global_model = None
_global_tokenizer = None
_global_vocab = None


def _initialize_globals():
    global _global_model, _global_tokenizer, _global_vocab

    if _global_tokenizer is None or _global_vocab is None:
        logger.info(f"Loading tokenizer: {TOKENIZER}")
        _global_tokenizer, _global_vocab = load_tokenizer_vocab(TOKENIZER)

    if _global_model is None:
        logger.info(f"Loading model from: {MODEL_PATH}")
        _global_model = load_model(
            MODEL_PATH,
            MODEL_NAME,
            len(_global_vocab['itos']),
            DEVICE,
            _global_vocab['stoi']['[PAD]'],
        )
        logger.info("Model and tokenizer loaded successfully")


def load_model(
    model_path: str, model_name: str, vocab_size: int, device: torch.device, padding_idx: int
) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == "transformer":
        model = Transformer(vocab_size, padding_idx)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = OrderedDict()

        for name, tensor in checkpoint['model_state_dict'].items():
            trunc_name = name
            if name.startswith("_orig_mod."):
                trunc_name = name[10:]

            state_dict[trunc_name] = tensor

        checkpoint['model_state_dict'] = state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return model
    except (RuntimeError, KeyError, OSError) as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def load_tokenizer_vocab(tokenizer_name: str) -> tuple[TweetTokenizer | BPETokenizer, dict]:
    if tokenizer_name == "tweet":
        tokenizer = TweetTokenizer()
    elif tokenizer_name == "bpe":
        tokenizer = BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

    vocab = tokenizer.get_vocab
    if vocab is None:
        raise ValueError(f"No vocabulary found for {tokenizer_name}")

    return tokenizer, vocab


def evaluate(
    model_path: str, model_name: str, tokenizer_name: str
) -> tuple[float, float, float, float, np.ndarray]:
    set_seed(SEED)

    tokenizer, vocab = load_tokenizer_vocab(tokenizer_name)

    test_path = os.path.join(DATA_PATH, "test.csv")
    target_mapping = {0: 0, 4: 1}

    test_loader = get_dataloaders(
        test_path,
        tokenizer,
        batch_size=1,
        target_mapping=target_mapping,
        is_train=False,
        max_length=MAX_LENGTH,
    )

    model = load_model(model_path, model_name, len(vocab['itos']), DEVICE, vocab['stoi']['[PAD]'])

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1: {f1:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info("\nTest Confusion Matrix:")
    logger.info(f"{cm}")

    return accuracy, f1, precision, recall, cm


def inference(text: str) -> tuple[str, float]:
    global _global_model, _global_tokenizer, _global_vocab

    _initialize_globals()

    set_seed(SEED)
    cleaned_text = TextCleaner.clean_text(text)

    encoding = _global_tokenizer.encode(cleaned_text, _global_vocab)
    input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        outputs = _global_model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = round(probabilities[0][prediction].item(), 3)

    return sentiment, confidence, cleaned_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="training/weights/transformer/tweet/0.pth")
    parser.add_argument("--model_name", default="transformer")
    parser.add_argument("--tokenizer", default="tweet")
    args = parser.parse_args()
    set_seed(SEED)

    evaluate(args.model_path, args.model_name, args.tokenizer)


if __name__ == "__main__":
    main()
