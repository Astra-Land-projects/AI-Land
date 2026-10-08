"""
Deploying an AI Model as a Web API (Flask)
Takes the Sentiment Analysis model and wraps it in a simple web API,
so it can be called from a browser, a mobile app, or any other program -
instead of only running inside a Python script.

Requires: pip install flask scikit-learn pandas
Requires: sentiment_dataset.csv (from the earlier sentiment analysis project)
          in the same folder. If missing, run generate_sentiment_dataset.py first.

Usage:
  python app.py
  Then open http://127.0.0.1:5000 in your browser, or send requests like:
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"I love this!\"}"
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# --- Train the model once when the server starts ---
print("Training model, please wait...")
df = pd.read_csv("sentiment_dataset.csv")
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print("Model ready!")

# --- A simple HTML page so you can test it in a browser ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis API</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 0 20px; }
        textarea { width: 100%; height: 100px; font-size: 16px; padding: 10px; }
        button { padding: 10px 20px; font-size: 16px; margin-top: 10px; cursor: pointer; }
        #result { margin-top: 20px; padding: 15px; border-radius: 8px; font-size: 18px; display: none; }
    </style>
</head>
<body>
    <h1>Sentiment Analysis API Demo</h1>
    <p>Type a review below to see the model predict its sentiment.</p>
    <textarea id="text" placeholder="e.g. This product exceeded my expectations!"></textarea>
    <br>
    <button onclick="analyze()">Analyze</button>
    <div id="result"></div>

    <script>
        async function analyze() {
            const text = document.getElementById('text').value;
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await response.json();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `<strong>Sentiment:</strong> ${data.sentiment.toUpperCase()}<br>
                <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%`;
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in the request body"}), 400

    text = data["text"]
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = max(probability)

    return jsonify({
        "text": text,
        "sentiment": prediction,
        "confidence": round(float(confidence), 4),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("\nStarting server at http://127.0.0.1:5000")
    print("Open that URL in your browser to try it out.\n")
    app.run(debug=True, port=5000)