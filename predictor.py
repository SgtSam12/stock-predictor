"""
predictor.py
ML prediction combining technical indicators (quantitative) with news
sentiment (qualitative) features.

Note on the sentiment feature during training: we only have a CURRENT
sentiment score (today's news), not historical daily sentiment for every
past trading day (that would require re-scoring news for every historical
date, which is expensive). So during training, the sentiment feature is
set to 0 (neutral) for historical rows, and the model still learns
primarily from price/indicator patterns. At PREDICTION time, we inject
today's real sentiment score. This is an honest limitation — sentiment
mainly nudges the live prediction and confidence, not the model's learned
weights. See README for how to extend this properly with a news archive.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score

TECHNICAL_FEATURES = [
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_mid",
    "volatility_20",
    "daily_return",
]
ALL_FEATURES = TECHNICAL_FEATURES + ["sentiment_score"]


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    data = df.copy()
    data["sentiment_score"] = 0.0  # neutral placeholder for historical rows — see module docstring
    data["target_price"] = data["close"].shift(-horizon)
    data["target_direction"] = (data["target_price"] > data["close"]).astype(int)

    cols_needed = TECHNICAL_FEATURES + ["target_price", "target_direction"]
    data = data.dropna(subset=cols_needed).reset_index(drop=True)
    return data


def train_direction_classifier(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = data[ALL_FEATURES]
    y = data["target_direction"]

    split_idx = int(len(data) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=random_state, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "baseline_accuracy": max(y_test.mean(), 1 - y_test.mean()),
        "n_test_samples": len(y_test),
    }
    feature_importance = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    return model, metrics, feature_importance


def train_price_regressor(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = data[ALL_FEATURES]
    y = data["target_price"]

    split_idx = int(len(data) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=random_state
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "mae_pct_of_price": mean_absolute_error(y_test, preds) / y_test.mean() * 100,
        "r2": r2_score(y_test, preds),
        "n_test_samples": len(y_test),
    }
    return model, metrics


def predict_latest(direction_model, price_model, full_indicator_df: pd.DataFrame, live_sentiment_score: float = 0.0) -> dict:
    """
    Predicts the next trading day using the most recent technical indicators
    PLUS a live sentiment score (from today's scored news).
    """
    latest = full_indicator_df.dropna(subset=TECHNICAL_FEATURES).iloc[[-1]].copy()
    latest["sentiment_score"] = live_sentiment_score
    X_latest = latest[ALL_FEATURES]

    direction_pred = direction_model.predict(X_latest)[0]
    direction_proba = direction_model.predict_proba(X_latest)[0]
    price_pred = price_model.predict(X_latest)[0]

    current_price = latest["close"].values[0]

    return {
        "current_price": float(current_price),
        "predicted_direction": "UP" if direction_pred == 1 else "DOWN",
        "confidence": float(max(direction_proba)),
        "predicted_next_price": float(price_pred),
        "predicted_change_pct": float((price_pred - current_price) / current_price * 100),
        "sentiment_score_used": live_sentiment_score,
    }
