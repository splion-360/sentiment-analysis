import argparse
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import wandb

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import (
    BATCH_SIZE,
    DATA_PATH,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    SEED,
    TOKENIZER,
    WANDB_PROJECT,
    set_seed,
    setup_logger,
)
from data.encoders import BPETokenizer, GPT2Tokenizer, TweetTokenizer
from data.utils import get_dataloaders
from training.models.transformer import Transformer

logger = setup_logger(__name__, "BLUE")

torch.set_float32_matmul_precision('high')


class EarlyStopping:

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def can_stop(self, loss: float) -> bool:

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info("Triggering Early Stopping")
                return True
            return False


class CosineDecayLR:
    def __init__(self, optimizer, T_max, lr_init, lr_min=0.0, warmup=0):
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup

    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (
                1 + np.cos(t / T_max * np.pi)
            )
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr


class Trainer:

    AVAILABLE_MODELS = ["transformer"]
    AVAILABLE_TOKENIZERS = ["tweet", "bpe", "gpt2"]

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        data_path: str,
        save_freq: int = 5,
        max_len: int = None,
        resume_path=None,
        use_wandb: bool = True,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
    ):
        set_seed(SEED)

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.save_freq = save_freq
        self.resume_path = resume_path
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.max_len = max_len
        self.use_wandb = use_wandb
        self.early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_delta)

        timestamp = int(time() * 1e7)

        self.save_path = f"training/weights/{self.model_name}/{self.tokenizer_name}/run_{timestamp}"
        os.makedirs(self.save_path, exist_ok=True)

        self._setup_tokenizer()
        self._setup_data_and_model()

        self.wandb_run_id = None
        if self.use_wandb:
            wandb.login()
            if self.resume_path and os.path.exists(self.resume_path):
                checkpoint = torch.load(self.resume_path, map_location='cpu', weights_only=False)
                self.wandb_run_id = checkpoint.get('wandb_run_id')

            self.run = wandb.init(
                project=WANDB_PROJECT,
                id=self.wandb_run_id,
                resume="allow" if self.wandb_run_id else None,
                config={
                    "dataset": "Sentiment Analysis (Twitter)",
                    "model": model_name,
                    "tokenizer": tokenizer_name,
                    "vocab_size": len(self.vocab['itos']),
                    "max_length": self.max_len,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "epochs": NUM_EPOCHS,
                },
                name=f"Training {model_name} with {tokenizer_name}",
            )
            self.wandb_run_id = self.run.id
            logger.info(f"Wandb run ID: {self.wandb_run_id}")

    def _setup_tokenizer(self):
        if self.tokenizer_name == "tweet":
            self.tokenizer = TweetTokenizer()
        elif self.tokenizer_name == "bpe":
            self.tokenizer = BPETokenizer()
        elif self.tokenizer_name == "gpt2":
            self.tokenizer = GPT2Tokenizer()
        else:
            raise ValueError(f"Unknown tokenizer: {self.tokenizer_name}")

        self.vocab = self.tokenizer.fetch_vocab
        if self.vocab is None:
            raise ValueError(
                f"No vocabulary found for {self.tokenizer_name}. Run prepare.py first."
            )

    def _setup_data_and_model(self):
        target_mapping = {0: 0, 4: 1}
        self.train_loader, self.val_loader = get_dataloaders(
            self.data_path,
            self.tokenizer,
            batch_size=BATCH_SIZE,
            target_mapping=target_mapping,
            stratify=True,
            max_length=self.max_len,
        )

        if self.model_name == "transformer":
            pad_idx = self.vocab['stoi']['[PAD]']
            self.model = Transformer(len(self.vocab['itos']), pad_idx)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        logger.info(f"Model initialized: {self.model_name}")
        logger.info(f"Tokenizer: {self.tokenizer_name}")
        logger.info(f"Vocab size: {len(self.vocab['itos'])}")

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load tokenizer
        checkpoint_tokenizer = checkpoint.get('tokenizer_name')
        if checkpoint_tokenizer is not None and checkpoint_tokenizer != self.tokenizer_name:
            logger.warning(
                f"Checkpoint tokenizer ({checkpoint_tokenizer}) \
                    differs from current ({self.tokenizer_name})"
            )
            logger.info(f"Using checkpoint tokenizer: {checkpoint_tokenizer}")
            self.tokenizer_name = checkpoint_tokenizer
            self._setup_tokenizer()

        if self.use_wandb:
            self.wandb_run_id = checkpoint.get('wandb_run_id')
            if self.wandb_run_id:
                logger.info(f"Will resume wandb run: {self.wandb_run_id}")

        logger.info(f"Resumed from epoch {self.start_epoch}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return optimizer

    def _process_model_output(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def train_epoch(self, epoch, optimizer, criterion, device, scheduler, step_offset):
        """Execute one training epoch with gradient updates and logging.

        Args:
            epoch: Current epoch number for logging
            optimizer: PyTorch optimizer for gradient updates
            criterion: Loss function for training
            device: Device to run training on
            scheduler: Learning rate scheduler
            step_offset: Global step offset for scheduler

        Returns:
            Tuple of (average_loss, predictions_list, labels_list)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for i, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            model_output = self._process_model_output(outputs)
            loss = criterion(model_output, labels)
            loss.backward()
            optimizer.step()

            scheduler.step(step_offset + i)

            total_loss += loss.item()
            _, predicted = torch.max(model_output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if i % LOG_INTERVAL == 0:
                current_lr = optimizer.param_groups[0]['lr']

                if self.use_wandb:
                    wandb.log({"batch_loss": loss.item()})

                logger.info(
                    f'Epoch: [{epoch} | {NUM_EPOCHS-1}]    '
                    f'Batch: [{i} | {len(self.train_loader)-1}]   '
                    f'loss: {loss.item():.3f}    '
                    f'lr: {current_lr:.6f}'
                )

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
                model_output = self._process_model_output(outputs)
                loss = criterion(model_output, labels)

                total_loss += loss.item()
                _, predicted = torch.max(model_output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return total_loss / len(self.val_loader), all_preds, all_labels

    def calculate_metrics(self, preds, labels):
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        cr = classification_report(labels, preds)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': cr,
        }

    def save_model(self, epoch, val_loss, optimizer, is_best=False):
        """Save model checkpoint with optional best model flag.

        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            optimizer: Current optimizer state
            is_best: Whether this is the best model (saves as best_model.pth)

        Returns:
            None
        """
        if is_best:
            model_path = os.path.join(self.save_path, "best_model.pth")
        else:
            model_path = os.path.join(self.save_path, f"{epoch}.pth")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'vocab': self.vocab,
            'tokenizer_name': self.tokenizer_name,
            'model_name': self.model_name,
            'wandb_run_id': self.wandb_run_id if self.use_wandb else None,
        }
        torch.save(checkpoint, model_path)

        if is_best:
            logger.info(f"Best model saved at {model_path}")
        else:
            logger.info(f"Model saved at {model_path}")

    def train(self, epochs, lr, device):
        """Train the model with cosine learning rate decay and early stopping.

        Args:
            epochs: Number of training epochs
            lr: Initial learning rate
            device: Device to train on (cuda/cpu)

        Returns:
            None
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model = torch.compile(self.model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        total_steps = epochs * len(self.train_loader)
        scheduler = CosineDecayLR(
            optimizer, T_max=total_steps, lr_init=lr, lr_min=1e-6, warmup=int(0.1 * total_steps)
        )

        if self.resume_path:
            optimizer = self.load_checkpoint(self.resume_path, optimizer)

        logger.info(f"Starting training for {epochs} epochs on {device}")
        logger.info(f"Training will run from epoch {self.start_epoch} to {epochs}")

        step = 0
        for epoch in range(self.start_epoch, epochs):
            train_loss, train_preds, train_labels = self.train_epoch(
                epoch, optimizer, criterion, device, scheduler, step
            )
            val_loss, val_preds, val_labels = self.validate_epoch(criterion, device)

            step += len(self.train_loader)

            train_metrics = self.calculate_metrics(train_preds, train_labels)
            val_metrics = self.calculate_metrics(val_preds, val_labels)

            current_lr = optimizer.param_groups[0]['lr']

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr,
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

            logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            logger.info(
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            logger.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")

            logger.info(f"\nTrain Confusion Matrix (Epoch {epoch}):")
            logger.info(f"{train_metrics['confusion_matrix']}")
            logger.info(f"\nValidation Confusion Matrix (Epoch {epoch}):")
            logger.info(f"{val_metrics['confusion_matrix']}")

            if epoch % self.save_freq == 0:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                    self.save_model(epoch, val_loss, optimizer, is_best=True)
                else:
                    self.save_model(epoch, val_loss, optimizer, is_best=False)

            if self.early_stopping.can_stop(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.save_model(epochs - 1, val_loss, optimizer)  # Save the model at the final epoch
        logger.info(f"Final model saved at epoch {epochs - 1}")

        if self.use_wandb:
            self.run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(DATA_PATH, "train.csv"),
        help="Training data path",
    )
    parser.add_argument("--save_freq", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    parser.add_argument("--wandb", action="store_true", default=False, help="Use wandb for logging")
    parser.add_argument(
        "--model", 
        type=str, 
        default=MODEL_NAME, 
        choices=Trainer.AVAILABLE_MODELS,
        help="Model architecture to use"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=TOKENIZER,
        choices=Trainer.AVAILABLE_TOKENIZERS,
        help="Tokenizer to use"
    )
    args = parser.parse_args()

    set_seed(SEED)

    trainer = Trainer(
        args.model,
        args.tokenizer,
        args.data_path,
        args.save_freq,
        MAX_LENGTH,
        args.resume,
        args.wandb,
        EARLY_STOPPING_PATIENCE,
        EARLY_STOPPING_MIN_DELTA,
    )
    trainer.train(NUM_EPOCHS, LEARNING_RATE, DEVICE)


if __name__ == "__main__":
    main()
