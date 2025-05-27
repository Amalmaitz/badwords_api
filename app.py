from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

@app.route('/')
def hi():
    return "hi"

@app.route('/receive', methods=['POST'])
def receive_words():
    data = request.get_json()
    words = data.get('words', [])
    
    # ✅ Print input words
    print("✅ Received words:", words)

    # Predict offensive words
    X = vectorizer.transform(words)
    predictions = model.predict(X)

    # ✅ Filter only offensive words (True)
    offensive_words = [word for word, pred in zip(words, predictions) if pred == 1]

    # ✅ Print result
    print("🧠 Offensive words:", offensive_words)

    return jsonify({
        "status": "success",
        "words": offensive_words  # ✅ Returning list of offensive words
    })

if __name__ == '__main__':
    app.run(debug=True)
