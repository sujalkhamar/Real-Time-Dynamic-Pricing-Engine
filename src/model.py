import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MODEL_FEATURES = ["lead_time", "total_guests", "stay_length"]
TARGET_COLUMN = "adr"


def build_model():
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )


def train_model(df):
    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    metrics = {
        "mae": round(mean_absolute_error(y_test, predictions), 2),
        "rmse": round(rmse, 2),
        "r2": round(r2_score(y_test, predictions), 3),
    }

    feature_importance = (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifacts = {
        "model": model,
        "features": MODEL_FEATURES,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }

    joblib.dump(artifacts, "model.pkl")
    return artifacts


def load_model_artifacts(path="model.pkl"):
    saved_object = joblib.load(path)

    if isinstance(saved_object, dict) and "model" in saved_object:
        return saved_object

    return {
        "model": saved_object,
        "features": MODEL_FEATURES,
        "metrics": None,
        "feature_importance": None,
    }
