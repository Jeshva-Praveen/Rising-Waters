from pathlib import Path

from flask import Flask, render_template, request
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

model = joblib.load(MODEL_DIR / "flood_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    rainfall = float(request.form["Rainfall"])
    cloud = float(request.form["CloudVisibility"])
    seasonal = float(request.form["SeasonalRainfall"])
    temperature = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])

    data = np.array([[rainfall, cloud, seasonal, temperature, humidity]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "⚠ HIGH FLOOD RISK"
    else:
        result = "✅ NO FLOOD RISK"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    print("\n====================================================")
    print("              RISING WATERS APP")
    print("====================================================")
    print("Flask App Local : http://127.0.0.1:8000")
    print("Vercel App      : https://rising-waters-liard.vercel.app")
    print("====================================================")
    app.run(debug=False, host="0.0.0.0", port=8000)