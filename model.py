import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# Create training dataset
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "power_sent": np.random.uniform(100, 500, n),
    "temperature": np.random.uniform(15, 45, n),
    "line_length_km": np.random.uniform(1, 50, n),
    "transformer_load": np.random.uniform(40, 100, n),
})

data["power_loss"] = (
    0.02 * data["line_length_km"]
    + 0.03 * data["temperature"]
    + 0.04 * data["transformer_load"]
    + np.random.normal(0, 2, n)
)

X = data.drop("power_loss", axis=1)
y = data["power_loss"]

# Train prediction model
loss_model = RandomForestRegressor()
loss_model.fit(X, y)

# Train anomaly model
data["predicted"] = loss_model.predict(X)
data["gap"] = abs(data["power_loss"] - data["predicted"])

anomaly_model = IsolationForest(contamination=0.05)
anomaly_model.fit(data[["gap"]])

def predict_loss(input_data):
    df = pd.DataFrame([input_data])
    pred = loss_model.predict(df)[0]
    return round(pred,2)

def detect_anomaly(input_data, actual_loss):
    df = pd.DataFrame([input_data])
    pred = loss_model.predict(df)[0]
    gap = abs(actual_loss - pred)
    result = anomaly_model.predict([[gap]])
    return "⚠️ Possible Theft/Fault" if result[0] == -1 else "✅ Normal"