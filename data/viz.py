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


def calculate_oov_rate(texts, tokenizer, vocab):
    total_tokens = 0
    oov_tokens = 0
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            total_tokens += 1
            if token not in vocab['stoi']:
                oov_tokens += 1
    return (oov_tokens / total_tokens * 100) if total_tokens > 0 else 0

def validate_tokenizer(texts, tokenizer_name, tokenizer, vocab):
    logger.info(f"\n=== {tokenizer_name.upper()} TOKENIZER ANALYSIS ===")
    
    oov_rate = calculate_oov_rate(texts, tokenizer, vocab)
    logger.info(f"Out-of-Vocabulary Rate: {oov_rate:.3f}%")
    
    patterns = {
        'URLs': [t for t in texts if 'http' in t.lower()],
        'Mentions': [t for t in texts if '@' in t],
        'Hashtags': [t for t in texts if '#' in t],
        'Misspellings': [t for t in texts if any(word in t.lower() for word in ['sooo', 'loooove', 'hahaha', 'yesss', 'nooo'])],
        'Emoticons': [t for t in texts if any(emo in t for emo in [':)', ':(', ':D', ':P', '=)'])],
        'Abbreviations': [t for t in texts if any(abbr in t.upper() for abbr in ['LOL', 'OMG', 'BTW', 'TBH', 'SMH'])]
    }
    
    logger.info(f"\nPattern-specific OOV rates:")
    for pattern_name, pattern_texts in patterns.items():
        if pattern_texts:
            pattern_oov = calculate_oov_rate(pattern_texts[:100], tokenizer, vocab)  # Limit for performance
            logger.info(f"  {pattern_name}: {pattern_oov:.3f}% (from {len(pattern_texts)} samples)")
    
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(len(tokenizer.tokenize(text)) for text in texts)
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    
    logger.info(f"\nEfficiency Metrics:")
    logger.info(f"  Characters per token: {chars_per_token:.2f}")
    logger.info(f"  Compression ratio: {total_chars}/{total_tokens} = {total_chars/total_tokens:.2f}")
    
    return {
        'tokenizer': tokenizer_name,
        'oov_rate': oov_rate,
        'chars_per_token': chars_per_token,
        'compression_ratio': total_chars/total_tokens if total_tokens > 0 else 0,
        'total_tokens': total_tokens,
        'vocab_size': len(vocab['itos'])
    }
