"""
Sentiment Analysis (Positive / Negative / Neutral)
Uses TF-IDF text vectorization + Logistic Regression / Naive Bayes classifiers.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path="sentiment_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} reviews")
    print(df["sentiment"].value_counts())
    return df


def prepare_data(df):
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = ["negative", "neutral", "positive"]
    print(f"Labels order: {labels}")
    print(confusion_matrix(y_test, predictions, labels=labels))

    return model, accuracy


def classify_review(model, vectorizer):
    print("\n" + "=" * 40)
    print("Classify a new review")
    print("=" * 40)

    review = input("Enter a review to analyze: ").strip()
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    classes = model.classes_
    prob_dict = dict(zip(classes, probability))

    print(f"\nPredicted sentiment: {prediction.upper()}")
    print("Confidence breakdown:")
    for cls, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.1%}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)

    nb_model, nb_acc = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        MultinomialNB(), "Naive Bayes"
    )

    lr_model, lr_acc = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        LogisticRegression(max_iter=1000), "Logistic Regression"
    )

    print("\n" + "=" * 40)
    if lr_acc >= nb_acc:
        print("Logistic Regression performed better (or equal) ✅")
        best_model = lr_model
    else:
        print("Naive Bayes performed better ✅")
        best_model = nb_model

    while True:
        choice = input("\nWould you like to analyze a new review? (yes/no): ").strip().lower()
        if choice == "yes":
            classify_review(best_model, vectorizer)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()