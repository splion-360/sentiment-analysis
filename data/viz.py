import matplotlib.pyplot as plt
import numpy as np

from config import LABELS, setup_logger

logger = setup_logger(__name__)


def get_histogram(data: np.ndarray, **kwargs):

    if not data:
        logger.error("Cannot create histogram from empty array")
        return

    title = kwargs.get("title", "train")
    labels, counts = np.unique(data, return_counts=True)
    fig, ax = plt.subplots()
    ax.set_title(f"Class Distribution: {title}")
    ax.bar(labels, counts, align='edge')
    if LABELS is not None:
        ax.set_xticks(labels)
        ax.set_xticklabels([LABELS[idx] for idx in labels])
    ax.set_ylabel("COUNT")
    ax.set_xlabel("CLASS")
    return fig, ax
