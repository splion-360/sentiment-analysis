import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import BATCH_SIZE, DATA_PATH, TEST_FILE, setup_logger

logger = setup_logger(__name__)
from data.tokenizers import BPETokenizer, TweetTokenizer
from data.utils import SentimentDataset, construct_data
from training.models.transformer import Transformer


def load_model(model_path, model_name, vocab_size, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == "transformer":
        model = Transformer(vocab_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model
    except (RuntimeError, KeyError, OSError) as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def load_tokenizer_vocab(tokenizer_name):
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


def evaluate(model_path, model_name, tokenizer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, vocab = load_tokenizer_vocab(tokenizer_name)

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    test_data = construct_data(test_path)

    target_mapping = {0: 0, 2: 1, 4: 1}
    test_dataset = SentimentDataset(test_data, tokenizer, vocab, target_mapping=target_mapping)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(model_path, model_name, len(vocab['itos']), device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def inference(text, model_path, model_name, tokenizer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, vocab = load_tokenizer_vocab(tokenizer_name)
    model = load_model(model_path, model_name, len(vocab['itos']), device)

    encoding = tokenizer.encode(text, vocab)
    input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[0][prediction].item()

    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")

    return sentiment, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "inference"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="transformer")
    parser.add_argument("--tokenizer", default="tweet")
    parser.add_argument("--text", help="Text for inference mode")
    args = parser.parse_args()

    if args.mode == "evaluate":
        evaluate(args.model_path, args.model_name, args.tokenizer)
    elif args.mode == "inference":
        if not args.text:
            raise ValueError("Text required for inference mode")
        inference(args.text, args.model_path, args.model_name, args.tokenizer)


if __name__ == "__main__":
    main()
