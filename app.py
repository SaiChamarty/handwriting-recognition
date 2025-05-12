# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np

# ——— 1. Load your trained parameters ———
# (Make sure you’ve saved W.npy and b.npy after training)
W = np.load("W.npy")   # shape (784,10)
b = np.load("b.npy")   # shape (1,10)

# ——— 2. Define helper functions ———
def compute_logits(X, W, b):
    return X.dot(W) + b

def softmax(logits):
    shifted    = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# ——— 3. Create the Flask app ———
app = Flask(__name__)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like { "pixels": [p0, p1, ..., p783] }
    where each p is a float in [0,1].
    """
    data = request.get_json()
    pixels = np.array(data["pixels"], dtype=np.float32)
    # Ensure shape (1, 784)
    x = pixels.reshape(1, -1)

    # Run the model
    logits = compute_logits(x, W, b)
    probs  = softmax(logits)
    pred   = int(np.argmax(probs, axis=1)[0])
    conf   = float(probs[0, pred])

    return jsonify({
        "prediction": pred,
        "confidence": conf
    })

if __name__ == "__main__":
    # For development; in production, use a WSGI server like gunicorn
    app.run(host="0.0.0.0", port=3000, debug=True)