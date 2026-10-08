"""
Generate a synthetic dataset of product/movie reviews labeled
as positive, negative, or neutral.
For a real project, you could use datasets like IMDB Reviews or Amazon Reviews.
"""

import pandas as pd
import random

random.seed(42)

positive_templates = [
    "This {item} is absolutely amazing, I love it!",
    "Best {item} I've ever bought, highly recommend!",
    "Great quality and works perfectly. Very happy with this {item}.",
    "Exceeded my expectations, this {item} is fantastic.",
    "Excellent {item}, worth every penny.",
    "I'm so impressed with this {item}, five stars!",
    "Wonderful experience, this {item} made my day.",
    "Superb quality, will definitely buy this {item} again.",
    "Perfect! Exactly what I needed, love this {item}.",
    "Really happy with this purchase, the {item} is great.",
]

negative_templates = [
    "This {item} is terrible, complete waste of money.",
    "Very disappointed with this {item}, would not recommend.",
    "Poor quality, the {item} broke after one use.",
    "Worst {item} I've ever purchased, total garbage.",
    "Awful experience, this {item} does not work at all.",
    "Not worth the price, this {item} is a disappointment.",
    "I regret buying this {item}, it's completely useless.",
    "Bad quality and terrible customer service for this {item}.",
    "This {item} stopped working within a week, very frustrating.",
    "Extremely unhappy with this {item}, asking for a refund.",
]

neutral_templates = [
    "The {item} is okay, nothing special but does the job.",
    "Average {item}, meets basic expectations.",
    "It's a decent {item}, not great but not bad either.",
    "The {item} works as described, nothing more to say.",
    "Fair quality {item} for the price, an average experience.",
    "The {item} is fine, I have no strong opinion either way.",
    "It does what it's supposed to, an ordinary {item}.",
    "Not bad, not amazing, just a standard {item}.",
    "The {item} arrived on time and functions normally.",
    "A reasonable {item}, met my basic needs.",
]

items = ["product", "phone", "laptop", "movie", "book", "restaurant",
         "headphones", "service", "app", "game"]


def fill_template(template):
    return template.format(item=random.choice(items))


def generate_dataset(n_per_class=150):
    rows = []
    for _ in range(n_per_class):
        rows.append({"review": fill_template(random.choice(positive_templates)), "sentiment": "positive"})
        rows.append({"review": fill_template(random.choice(negative_templates)), "sentiment": "negative"})
        rows.append({"review": fill_template(random.choice(neutral_templates)), "sentiment": "neutral"})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("sentiment_dataset.csv", index=False)
    print(f"Dataset created: sentiment_dataset.csv ({len(df)} reviews)")
    print(df.head())
    print("\nLabel distribution:")
    print(df["sentiment"].value_counts())