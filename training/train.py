import argparse
import os
import sys

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_EPOCHS, WANDB_PROJECT, setup_logger
from data.tokenizers import BPETokenizer, TweetTokenizer
from data.utils import SentimentDataset, construct_data, split_data
from training.models.transformer import Transformer

logger = setup_logger(__name__, "BLUE")


class Trainer:

    AVAILABLE_MODELS = ["transformer"]
    AVAILABLE_TOKENIZERS = ["tweet", "bpe"]

    def __init__(self, model_name, tokenizer_name, data_path, save_freq=5, resume_path=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.save_freq = save_freq
        self.resume_path = resume_path
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        os.makedirs(f"training/weights/{model_name}", exist_ok=True)

        self._setup()

    def _setup(self):
        if self.tokenizer_name == "tweet":
            self.tokenizer = TweetTokenizer()
        elif self.tokenizer_name == "bpe":
            self.tokenizer = BPETokenizer()
        else:
            raise ValueError(f"Unknown tokenizer: {self.tokenizer_name}")

        self.vocab = self.tokenizer.get_vocab
        if self.vocab is None:
            data = construct_data(self.data_path)
            self.vocab = self.tokenizer.build_vocab(data['cleaned_text'].tolist())

        data = construct_data(self.data_path)
        train_df, val_df = split_data(data)

        target_mapping = {0: 0, 4: 1}
        train_dataset = SentimentDataset(
            train_df, self.tokenizer, self.vocab, target_mapping=target_mapping
        )
        val_dataset = SentimentDataset(
            val_df, self.tokenizer, self.vocab, target_mapping=target_mapping
        )

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        if self.model_name == "transformer":
            self.model = Transformer(len(self.vocab['itos']))
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        logger.info(f"Model initialized: {self.model_name}")
        logger.info(f"Tokenizer: {self.tokenizer_name}")
        logger.info(f"Vocab size: {len(self.vocab['itos'])}")

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' not in checkpoint:
                logger.error("Invalid checkpoint format: missing model_state_dict")
                raise KeyError("Invalid checkpoint format: missing model_state_dict")

            self.model.load_state_dict(checkpoint['model_state_dict'])

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            logger.info(f"Resumed from epoch {self.start_epoch}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

            return optimizer
        except (RuntimeError, KeyError) as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to read checkpoint file {checkpoint_path}: {e}")
            raise

    def train_epoch(self, optimizer, criterion, device):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return total_loss / len(self.train_loader), all_preds, all_labels

    def validate_epoch(self, criterion, device):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return total_loss / len(self.val_loader), all_preds, all_labels

    def calculate_metrics(self, preds, labels):
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')

        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    def save_model(self, epoch, val_loss, optimizer):
        model_path = f"training/weights/{self.model_name}/{epoch}.pth"

        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss,
                'vocab': self.vocab,
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved at {model_path}")
        except OSError as e:
            logger.error(f"Failed to save model to {model_path}: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"PyTorch error saving model: {e}")
            raise

    def train(self, epochs, lr, device):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if self.resume_path:
            optimizer = self.load_checkpoint(self.resume_path, optimizer)

        logger.info(f"Starting training for {epochs} epochs on {device}")
        logger.info(f"Training will run from epoch {self.start_epoch} to {epochs}")

        for epoch in range(self.start_epoch, epochs):
            train_loss, train_preds, train_labels = self.train_epoch(optimizer, criterion, device)
            val_loss, val_preds, val_labels = self.validate_epoch(criterion, device)

            train_metrics = self.calculate_metrics(train_preds, train_labels)
            val_metrics = self.calculate_metrics(val_preds, val_labels)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_metrics['accuracy'],
                    "train_f1": train_metrics['f1'],
                    "train_precision": train_metrics['precision'],
                    "train_recall": train_metrics['recall'],
                    "val_accuracy": val_metrics['accuracy'],
                    "val_f1": val_metrics['f1'],
                    "val_precision": val_metrics['precision'],
                    "val_recall": val_metrics['recall'],
                }
            )

            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            logger.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")

            if epoch % self.save_freq == 0 or val_loss < self.best_val_loss:
                self.save_model(epoch, val_loss, optimizer)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=Trainer.AVAILABLE_MODELS)
    parser.add_argument("--tokenizer", default="tweet", choices=Trainer.AVAILABLE_TOKENIZERS)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    wandb.init(project=WANDB_PROJECT, config=vars(args))

    trainer = Trainer(args.model, args.tokenizer, args.data_path, args.save_freq, args.resume)
    trainer.train(args.epochs, args.lr, args.device)

    wandb.finish()


if __name__ == "__main__":
    main()
