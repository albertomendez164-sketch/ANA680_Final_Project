from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path("wine_quality_model.pkl")
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "Model file not found. Run python train_model.py first."
        else:
            try:
                values = {feature: float(request.form[feature]) for feature in FEATURES}
                sample = pd.DataFrame([values], columns=FEATURES)
                pred = float(model.predict(sample)[0])
                pred = max(0.0, min(10.0, pred))
                prediction = round(pred, 2)
            except Exception as exc:
                error = f"Prediction error: {exc}"

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        error=error
    )

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}, 200

if __name__ == "__main__":
    app.run(debug=True)
