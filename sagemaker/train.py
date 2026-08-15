"""Starter SageMaker training script for the ANA680 final project.

Upload/copy the dataset into the SageMaker training channel and adapt
the input path to the container environment used in your Studio Lab.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
data_path = os.path.join(train_dir, "winequality-red.csv")

df = pd.read_csv(data_path, sep=";").drop_duplicates().dropna()
X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** 0.5
print(f"RMSE: {rmse:.4f}")

os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
