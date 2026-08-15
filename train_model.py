from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/winequality-red.csv")
MODEL_PATH = Path("wine_quality_model.pkl")
METRICS_PATH = Path("model_metrics.json")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Place winequality-red.csv inside the data/ folder before running this script."
    )

df = pd.read_csv(DATA_PATH, sep=";")
df = df.drop_duplicates().dropna()

X = df.drop(columns=["quality"])
y = df["quality"]

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

validation_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    validation_results[name] = {
        "MAE": float(mean_absolute_error(y_val, pred)),
        "RMSE": float(mean_squared_error(y_val, pred) ** 0.5),
        "R2": float(r2_score(y_val, pred))
    }

best_name = min(validation_results, key=lambda n: validation_results[n]["RMSE"])
best_model = models[best_name]

# Refit selected algorithm using train + validation data.
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])
best_model.fit(X_train_final, y_train_final)

test_pred = best_model.predict(X_test)
test_metrics = {
    "MAE": float(mean_absolute_error(y_test, test_pred)),
    "RMSE": float(mean_squared_error(y_test, test_pred) ** 0.5),
    "R2": float(r2_score(y_test, test_pred))
}

joblib.dump(best_model, MODEL_PATH)

output = {
    "best_model": best_name,
    "split": {"training": 0.70, "validation": 0.15, "testing": 0.15},
    "validation_results": validation_results,
    "test_metrics": test_metrics,
    "rows_after_cleaning": int(len(df)),
    "features": list(X.columns)
}
METRICS_PATH.write_text(json.dumps(output, indent=2))

print("Best model:", best_name)
print("Test metrics:", test_metrics)
print("Saved:", MODEL_PATH)
print("Saved:", METRICS_PATH)
