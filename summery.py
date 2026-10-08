"""
Simple Text Summarizer (Extractive)
Scores each sentence using TF-IDF importance and picks the top N sentences.
No API needed, no internet required after install - lightweight and fast.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SAMPLE_TEXT = """
Artificial intelligence has transformed many industries over the past decade.
Machine learning models can now recognize images, understand language, and even
generate creative content. Companies across sectors are adopting AI to improve
efficiency and reduce costs. However, this rapid growth also raises important
questions about ethics, privacy, and job displacement. Researchers continue to
study how AI systems make decisions to ensure they are fair and transparent.
Governments around the world are beginning to draft regulations for AI
technologies. Education systems are also adapting, adding AI and data science
courses to prepare students for the future job market. Despite the challenges,
most experts agree that AI will continue to play a growing role in daily life.
The key to successful AI adoption lies in balancing innovation with responsible
use. As the technology matures, collaboration between industry, government, and
academia will be essential to navigate its risks and benefits.
"""


def split_sentences(text):
    """Splits text into sentences using simple punctuation-based rules."""
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences


def summarize(text, num_sentences=3):
    sentences = split_sentences(text)

    if len(sentences) <= num_sentences:
        return sentences  # nothing to summarize, text is already short

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score each sentence by the sum of its TF-IDF weights
    sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    # Pick indices of the top-scoring sentences, then restore original order
    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices]
    return summary_sentences


def print_summary(sentences):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for s in sentences:
        print(f"- {s}")


def main():
    print("=" * 50)
    print("Simple Text Summarizer")
    print("=" * 50)
    print("\n1. Use the built-in sample text")
    print("2. Paste your own text")
    choice = input("\nChoose an option (1/2): ").strip()

    if choice == "2":
        print("\nPaste your text below, then press Enter and type 'END' on a new line:")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        text = " ".join(lines)
    else:
        text = SAMPLE_TEXT
        print("\nUsing sample text about AI.")

    num_sentences = input("\nHow many sentences should the summary have? (default 3): ").strip()
    num_sentences = int(num_sentences) if num_sentences.isdigit() else 3

    original_sentences = split_sentences(text)
    summary = summarize(text, num_sentences)

    print(f"\nOriginal text: {len(original_sentences)} sentences")
    print(f"Summary: {len(summary)} sentences")

    print_summary(summary)


if __name__ == "__main__":
    main()