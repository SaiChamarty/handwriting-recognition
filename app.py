# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

# ——— 1. Load your trained CNN model ———
model = load_model("cnn_mnist_model.keras", compile=False)

# ——— 2. Create the Flask app ———
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # must have templates/index.html

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like { "pixels": [p0, p1, ..., p783] }
    where each p is a float in [0,1].
    """
    data = request.get_json()

    # 1. Convert to (28, 28)
    pixels = np.array(data["pixels"], dtype=np.float32).reshape(28, 28)

    # 2. Prepare input for CNN: (1, 28, 28, 1)
    img = pixels.reshape(1, 28, 28, 1)

    # 3. Predict
    probs = model.predict(img, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    return jsonify({
        "prediction": pred,
        "confidence": conf
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)