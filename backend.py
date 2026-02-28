from flask import Flask, render_template, request, jsonify
from model import predict_loss, detect_anomaly

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    input_data = {
        "power_sent": float(data.get("power_sent",0)),
        "temperature": float(data.get("temperature",0)),
        "line_length_km": float(data.get("line_length",0)),
        "transformer_load": float(data.get("load",0))
    }
    
    predicted = predict_loss(input_data)
    status = detect_anomaly(input_data, float(data.get("actual_loss",0)))
    
    return jsonify({
        "predicted_loss": predicted,
        "status": status
    })

if __name__ == "__main__":
    app.run(debug=True)