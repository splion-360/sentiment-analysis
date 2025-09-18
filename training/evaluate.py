import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
from data.encoders import BPETokenizer, GPT2Tokenizer, TweetTokenizer
from data.utils import TextCleaner, get_dataloaders
from training.models.transformer import Transformer

logger = setup_logger(__name__, "TURQUOISE")

_global_model = None
_global_tokenizer = None
_global_vocab = None


def _initialize_globals():
    global _global_model, _global_tokenizer, _global_vocab

    if _global_tokenizer is None or _global_vocab is None:
        tokenizer_name = TOKENIZER
        model_name = MODEL_NAME

        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
                checkpoint_tokenizer = checkpoint.get('tokenizer_name')
                checkpoint_model = checkpoint.get('model_name')

                if checkpoint_tokenizer is not None:
                    tokenizer_name = checkpoint_tokenizer
                    logger.info(f"Inferred tokenizer: {tokenizer_name} from checkpoint")

                if checkpoint_model is not None:
                    model_name = checkpoint_model
                    logger.info(f"Inferred model: {model_name} from checkpoint")

            except Exception as e:
                logger.warning(f"Could not infer from checkpoint, using config defaults: {e}")

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        _global_tokenizer, _global_vocab = load_tokenizer_vocab(tokenizer_name)

    if _global_model is None:
        logger.info(f"Loading model from: {MODEL_PATH}")
        _global_model = load_model(
            MODEL_PATH,
            model_name,
            len(_global_vocab['itos']),
            DEVICE,
            _global_vocab['stoi']['[PAD]'],
        )
        logger.info("Model and tokenizer loaded successfully")


def load_model(
    model_path: str, model_name: str, vocab_size: int, device: torch.device, padding_idx: int
) -> torch.nn.Module:
    """
    Load trained model from checkpoint

    Args:
        model_path: Path to model checkpoint file
        model_name: Name of model architecture to instantiate
        vocab_size: Size of vocabulary for model initialization
        device: Device to load model onto
        padding_idx: Padding token index for model initialization

    Returns:
        model: For inference
    """
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
            if name.startswith("_orig_mod."):  # torch.compile() modifies the keys (stripping reqd.)
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
    elif tokenizer_name == "gpt2":
        tokenizer = GPT2Tokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

    vocab = tokenizer.fetch_vocab
    if vocab is None:
        raise ValueError(f"No vocabulary found for {tokenizer_name}")

    return tokenizer, vocab


def evaluate(
    model_path: str, model_name: str = None, tokenizer_name: str = None
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Evaluate trained model and report metrics.

    Args:
        model_path: Path to saved model checkpoint
        model_name: Name of model architecture to load
        tokenizer_name: Name of tokenizer to use for text processing

    Returns:
        Tuple of (accuracy, f1_score, precision, recall, confusion_matrix, classification_report)
    """
    set_seed(SEED)

    if model_name is None or tokenizer_name is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if model_name is None:
            model_name = checkpoint.get('model_name')
            if model_name is None:
                raise ValueError("Model name not found in checkpoint")
            logger.info(f"Inferred model name from checkpoint: {model_name}")

        if tokenizer_name is None:
            tokenizer_name = checkpoint.get('tokenizer_name')
            if tokenizer_name is None:
                raise ValueError("Tokenizer name not found in checkpoint")
            logger.info(f"Inferred tokenizer name from checkpoint: {tokenizer_name}")

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
    cr = classification_report(all_labels, all_preds)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1: {f1:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info("\nTest Confusion Matrix:")
    logger.info(f"{cm}")
    logger.info("\nTest Classification Report:")
    logger.info(f"{cr}")

    return accuracy, f1, precision, recall, cm, cr


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
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    set_seed(SEED)

    evaluate(args.model_path)


if __name__ == "__main__":
    main()
