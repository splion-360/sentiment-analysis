import logging
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

LABELS = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

# Dataset & Tokenizer Config
DATA_PATH = 'data/trainingandtestdata/'
TRAIN_FILE = 'training.1600000.processed.noemoticon.csv'  # this is the original train file
# (do not use this for training)
TEST_FILE = 'testdata.manual.2009.06.14.csv'  # this is the unprocessed test file
# (do not use this for testing)
MAX_VOCAB_SIZE = 30000
MIN_FREQ = 2
MAX_LENGTH = 40
USE_STEMMING = True
SEED = 43


# TRANSFORMER
class TransformerConfig:
    EMBEDDING_DIM = 128
    NUM_HEADS = 4
    FF_PROJECTION_DIM = 512
    NUM_CLASSES = 2
    DROPOUT = 0.1
    LAYERS = 3


# Training Config
MODEL_NAME = "transformer"
TOKENIZER = "bpe"

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.99
DEVICE = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
WANDB_PROJECT = "sentiment-analysis-improved"
NUM_WORKERS = 16
LOG_INTERVAL = 1000
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-3


# Inference (App configs)
MODEL_PATH = "training/inference/best.pth"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "BLACK": "\033[30m",
        "RED": "\033[31m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "BLUE": "\033[34m",
        "MAGENTA": "\033[35m",
        "CYAN": "\033[36m",
        "VIOLET": "\033[38;5;213m",
        "TURQUOISE": "\033[38;5;80m",
        "CRIMSON": "\033[38;5;196m",
        "FOREST": "\033[38;5;28m",
        "ROYAL": "\033[38;5;63m",
        "AMBER": "\033[38;5;214m",
        "EMERALD": "\033[38;5;46m",
        "RESET": "\033[0m",
    }
    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __init__(self, color=None):
        super().__init__()
        self.color = color

        if color and color in self.COLORS:
            self.FORMATS = {
                logging.DEBUG: self.COLORS[color] + self.base_format + self.COLORS["RESET"],
                logging.INFO: self.COLORS[color] + self.base_format + self.COLORS["RESET"],
                logging.WARNING: self.COLORS[color] + self.base_format + self.COLORS["RESET"],
                logging.ERROR: self.COLORS[color] + self.base_format + self.COLORS["RESET"],
                logging.CRITICAL: self.COLORS[color] + self.base_format + self.COLORS["RESET"],
            }
        else:
            self.FORMATS = {
                logging.DEBUG: self.COLORS["YELLOW"] + self.base_format + self.COLORS["RESET"],
                logging.INFO: self.COLORS["BLUE"] + self.base_format + self.COLORS["RESET"],
                logging.WARNING: self.COLORS["AMBER"] + self.base_format + self.COLORS["RESET"],
                logging.ERROR: self.COLORS["RED"] + self.base_format + self.COLORS["RESET"],
                logging.CRITICAL: self.COLORS["ROYAL"] + self.base_format + self.COLORS["RESET"],
            }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


def setup_logger(name: str = __name__, text_color: str = None) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler()

        color = None
        if text_color and text_color.upper() in ColoredFormatter.COLORS:
            color = text_color.upper()

        custom_formatter = ColoredFormatter(color)
        console_handler.setFormatter(custom_formatter)
        logger.addHandler(console_handler)

    return logger


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


logger = setup_logger()
