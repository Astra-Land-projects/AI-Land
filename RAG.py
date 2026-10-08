"""
RAG (Retrieval-Augmented Generation) over a PDF
Ask questions about a PDF document in plain English. The script:
  1. Extracts and chunks the text from the PDF
  2. Finds the most relevant chunks for your question (TF-IDF similarity)
  3. Sends those chunks + your question to Claude to generate an answer

This is a simplified version of RAG - real production systems use vector
embeddings and a vector database instead of TF-IDF, but the core idea
(retrieve relevant context, then generate an answer from it) is the same.

Requires: pip install pypdf anthropic scikit-learn
Requires an Anthropic API key set as an environment variable:
  export ANTHROPIC_API_KEY="your-key-here"   (macOS/Linux)
  $env:ANTHROPIC_API_KEY="your-key-here"     (Windows PowerShell)

Usage: python rag_pdf.py your_document.pdf
"""

import os
import sys
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic

MODEL = "claude-sonnet-5"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks so context isn't lost at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def find_relevant_chunks(question, chunks, vectorizer, chunk_vectors, top_n=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Get a key at https://console.anthropic.com/settings/keys")
        exit(1)
    return Anthropic(api_key=api_key)


def ask_claude(client, question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the question based only on the following context from a document.
If the answer isn't in the context, say you don't know based on the document.

Context:
{context}

Question: {question}

Answer:"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_pdf.py <path_to_pdf>")
        return

    pdf_path = sys.argv[1]

    print(f"Reading {pdf_path}...")
    text = extract_text(pdf_path)
    print(f"Extracted {len(text)} characters.")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.")

    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunks)

    client = get_client()

    print("\nAsk questions about the document. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        if not question:
            continue

        relevant_chunks = find_relevant_chunks(question, chunks, vectorizer, chunk_vectors)

        try:
            answer = ask_claude(client, question, relevant_chunks)
        except Exception as e:
            print(f"Error calling the API: {e}")
            continue

        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()