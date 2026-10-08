"""
Language Detector
Detects the language of a short text using character n-gram TF-IDF
features + a Naive Bayes classifier. Character n-grams work well for
language detection because they capture letter patterns unique to
each language, even with a small dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data(path="language_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} sentences")
    print(df["language"].value_counts())
    return df


def prepare_data(df):
    X = df["text"]
    y = df["language"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Character n-grams (2 to 4 letters) capture language-specific patterns
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    return model


def detect_language(model, vectorizer):
    print("\n" + "=" * 40)
    print("Detect the language of a new sentence")
    print("=" * 40)

    text = input("Enter a sentence: ").strip()
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    classes = model.classes_
    prob_dict = dict(zip(classes, probability))

    print(f"\nDetected language: {prediction.upper()}")
    print("Confidence breakdown:")
    for lang, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {prob:.1%}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)

    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    while True:
        choice = input("\nWould you like to detect the language of a sentence? (yes/no): ").strip().lower()
        if choice == "yes":
            detect_language(model, vectorizer)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()