"""
Resume/CV Screener
Compares a resume against a job description and scores how well
they match, using TF-IDF text similarity. Also extracts which
important keywords from the job description are missing in the resume.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_JOB_DESCRIPTION = """
We are looking for a Data Analyst with strong skills in Python, SQL, and
data visualization. The ideal candidate has experience with pandas, numpy,
and building dashboards using Tableau or Power BI. Familiarity with machine
learning concepts and statistics is a plus. Strong communication skills
and the ability to work with cross-functional teams are required. A
bachelor's degree in a quantitative field is preferred.
"""

SAMPLE_RESUME = """
Experienced professional with a background in Python programming and
data analysis. Skilled in using pandas and numpy for data manipulation.
Built several dashboards using Power BI to present insights to stakeholders.
Strong problem-solving skills and experience working in agile teams.
Bachelor's degree in Computer Science.
"""


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_match_score(resume_text, job_text):
    documents = [clean_text(resume_text), clean_text(job_text)]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity, vectorizer


def find_missing_keywords(resume_text, job_text, top_n=10):
    """Finds important words from the job description that don't appear in the resume."""
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    job_tfidf = vectorizer.fit_transform([job_clean])
    feature_names = vectorizer.get_feature_names_out()
    scores = job_tfidf.toarray()[0]

    # Sort job keywords by importance (TF-IDF score)
    keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    resume_words = set(resume_clean.split())
    missing = [(word, score) for word, score in keyword_scores if word not in resume_words]

    return missing[:top_n]


def read_multiline_input(prompt):
    print(prompt)
    print("(paste your text, then type 'END' on a new line)")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return " ".join(lines)


def main():
    print("=" * 50)
    print("Resume / CV Screener")
    print("=" * 50)

    print("\n1. Use sample resume + job description")
    print("2. Paste my own resume and job description")
    choice = input("\nChoose an option (1/2): ").strip()

    if choice == "2":
        resume_text = read_multiline_input("\nPaste the RESUME text:")
        job_text = read_multiline_input("\nPaste the JOB DESCRIPTION text:")
    else:
        resume_text = SAMPLE_RESUME
        job_text = SAMPLE_JOB_DESCRIPTION
        print("\nUsing sample resume and job description.")

    score, _ = compute_match_score(resume_text, job_text)

    print("\n" + "=" * 50)
    print("MATCH RESULTS")
    print("=" * 50)
    print(f"Match score: {score:.1%}")

    if score >= 0.5:
        print("Assessment: Strong match ✅")
    elif score >= 0.3:
        print("Assessment: Moderate match ⚠️")
    else:
        print("Assessment: Weak match ❌")

    missing = find_missing_keywords(resume_text, job_text)
    if missing:
        print("\nImportant keywords from the job description missing in the resume:")
        for word, weight in missing:
            print(f"  - {word} (importance: {weight:.2f})")
    else:
        print("\nNo important keywords are missing - great coverage!")


if __name__ == "__main__":
    main()