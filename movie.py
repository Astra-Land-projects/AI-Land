"""
Movie Recommendation System (Content-Based Filtering)
Recommends movies similar to one you like, based on genre and
description text similarity - using TF-IDF and cosine similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="movies.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} movies")
    return df


def build_similarity_matrix(df):
    # Combine genre and description into one feature text per movie
    df["combined_features"] = df["genre"] + " " + df["description"]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix


def recommend(df, similarity_matrix, title, top_n=5):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        return None

    idx = matches.index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Skip the first result since it's the movie itself
    top_matches = similarity_scores[1:top_n + 1]

    recommendations = []
    for movie_idx, score in top_matches:
        recommendations.append({
            "title": df.iloc[movie_idx]["title"],
            "genre": df.iloc[movie_idx]["genre"],
            "similarity": score,
        })

    return recommendations


def list_movies(df):
    print("\nAvailable movies:")
    for i, title in enumerate(df["title"], 1):
        print(f"  {i}. {title}")


def main():
    df = load_data()
    similarity_matrix = build_similarity_matrix(df)

    list_movies(df)

    while True:
        title = input("\nEnter a movie title you like (or 'list' to see all, 'exit' to quit): ").strip()

        if title.lower() == "exit":
            print("Goodbye! 👋")
            break

        if title.lower() == "list":
            list_movies(df)
            continue

        recommendations = recommend(df, similarity_matrix, title)

        if recommendations is None:
            print(f"Movie '{title}' not found. Type 'list' to see available titles.")
            continue

        print(f"\nBecause you liked '{title}', you might also like:")
        for r in recommendations:
            print(f"  - {r['title']} ({r['genre']}) - similarity: {r['similarity']:.1%}")


if __name__ == "__main__":
    main()