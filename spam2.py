"""
Generate a small labeled dataset of spam / ham (not-spam) email messages.
For a real project, you'd use a real dataset (e.g. the SMS Spam Collection dataset),
but this synthetic set is enough to demonstrate the full pipeline.
"""

import pandas as pd
import random

random.seed(42)

spam_templates = [
    "Congratulations! You have won a {prize}. Click here to claim now!",
    "URGENT: Your account will be suspended. Verify your details immediately.",
    "You've been selected for a free {prize}. Limited time offer, act now!",
    "Get rich quick! Earn ${amount} per day working from home, no experience needed.",
    "FINAL NOTICE: Your payment is overdue. Click this link to avoid penalty.",
    "Hot singles in your area want to meet you! Click now.",
    "You are pre-approved for a loan of ${amount}. Apply today, no credit check!",
    "Claim your free {prize} now before it's too late! Limited stock available.",
    "WINNER!! As a valued customer you have been selected to receive a {prize}.",
    "Lose 20 pounds in a week with this one weird trick! Click to learn more.",
]

ham_templates = [
    "Hey, are we still on for lunch tomorrow at {time}?",
    "Please find attached the report for last {period}.",
    "Can you send me the meeting notes from {period}?",
    "Reminder: your appointment is scheduled for {time}.",
    "Thanks for your help with the project last week.",
    "Let's catch up sometime this {period}, it's been a while.",
    "The invoice for this {period} has been processed.",
    "Could you review the document before {time}?",
    "Happy birthday! Hope you have a great day.",
    "Just checking in to see how everything is going.",
]

prizes = ["iPhone", "vacation package", "gift card", "cash prize", "laptop"]
amounts = ["500", "1000", "2500", "5000"]
times = ["9 AM", "noon", "3 PM", "Friday", "next Monday"]
periods = ["month", "week", "quarter", "year"]


def fill_template(template):
    return template.format(
        prize=random.choice(prizes),
        amount=random.choice(amounts),
        time=random.choice(times),
        period=random.choice(periods),
    )


def generate_dataset(n_per_class=150):
    rows = []
    for _ in range(n_per_class):
        rows.append({"message": fill_template(random.choice(spam_templates)), "label": "spam"})
        rows.append({"message": fill_template(random.choice(ham_templates)), "label": "ham"})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("spam_dataset.csv", index=False)
    print(f"Dataset created: spam_dataset.csv ({len(df)} messages)")
    print(df.head())
    print("\nLabel distribution:")
    print(df["label"].value_counts())