import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud

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


def kde_distribution(lengths: list, **kwargs):
    if not lengths:
        logger.error("Cannot create KDE plot from empty lengths list")
        return

    title = kwargs.get("title", "Text Length Distribution")
    figsize = kwargs.get("figsize", (10, 6))
    color = kwargs.get("color", "blue")

    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(data=lengths, ax=ax, color=color, fill=True, alpha=0.3)

    # Add statistics as vertical lines
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)

    ax.axvline(mean_len, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_len:.3f}')
    ax.axvline(
        median_len, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_len:.3f}'
    )

    ax.set_title(title)
    ax.set_xlabel("Text Length")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    logger.info(f"KDE plot created for {len(lengths)} text samples")
    return fig, ax


def plot_wordclouds(positive_texts: list, negative_texts: list, **kwargs):
    if not positive_texts or not negative_texts:
        logger.error("Cannot create wordclouds: empty text lists provided")
        return

    figsize = kwargs.get("figsize", (15, 6))
    max_words = kwargs.get("max_words", 100)
    background_color = kwargs.get("background_color", "white")

    # Combine texts for each sentiment
    positive_text = " ".join(positive_texts)
    negative_text = " ".join(negative_texts)

    # Create WordCloud objects
    wordcloud_pos = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color=background_color,
        colormap='Greens',
    ).generate(positive_text)

    wordcloud_neg = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color=background_color,
        colormap='Reds',
    ).generate(negative_text)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot positive wordcloud
    axes[0].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0].set_title('Positive Sentiment Words', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Plot negative wordcloud
    axes[1].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1].set_title('Negative Sentiment Words', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    logger.info(
        f"Wordclouds created from {len(positive_texts)} positive \
            and {len(negative_texts)} negative texts"
    )
    return fig, axes
