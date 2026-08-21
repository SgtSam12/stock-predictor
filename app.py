"""
app.py
FastAPI backend for the DSE AI Predictor.

Endpoints:
  GET  /api/predict/{ticker}?company_name=...&start=...&end=...
       -> full pipeline: fetch price data, fetch+score news, train models,
          return prediction + technical signals + news summary as JSON

  GET  /api/history/{ticker}?start=...&end=...
       -> historical OHLCV + indicators for charting

  GET  /api/health
       -> simple health check

Run locally:
    uvicorn app:app --reload --port 8000

Deploy on Render: see README.md for step-by-step instructions.
"""

import os
from datetime import date, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import data_fetch
import indicators
import news_fetch
import sentiment
import predictor

app = FastAPI(title="DSE AI Predictor API")

# Allow the frontend (hosted elsewhere, e.g. Netlify) to call this API.
# Tighten allow_origins to your actual frontend URL before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _get_price_data(ticker: str, start: str, end: str, demo: bool):
    if demo:
        return data_fetch.generate_sample_data(days=500)
    try:
        return data_fetch.fetch_from_bdshare(ticker, start, end)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch DSE price data for '{ticker}': {e}. "
                   f"Try ?demo=true to test with synthetic data instead.",
        )


@app.get("/api/history/{ticker}")
def get_history(
    ticker: str,
    start: str = Query(default=None),
    end: str = Query(default=None),
    demo: bool = Query(default=False),
):
    start = start or str(date.today() - timedelta(days=365 * 2))
    end = end or str(date.today())

    df = _get_price_data(ticker, start, end, demo)
    df_ind = indicators.compute_all_indicators(df)
    df_ind["date"] = df_ind["date"].astype(str)

    return {
        "ticker": ticker,
        "rows": df_ind.fillna("").to_dict(orient="records"),
        "signals": indicators.latest_signal_summary(indicators.compute_all_indicators(df)),
    }


@app.get("/api/predict/{ticker}")
def predict(
    ticker: str,
    company_name: str = Query(default=None, description="Full company name, improves news search relevance"),
    start: str = Query(default=None),
    end: str = Query(default=None),
    demo: bool = Query(default=False),
    skip_news: bool = Query(default=False, description="Skip news/sentiment step (faster, quant-only prediction)"),
):
    try:
        start = start or str(date.today() - timedelta(days=365 * 3))
        end = end or str(date.today())
        company_name = company_name or ticker

        # 1. Price data + technical indicators
        df = _get_price_data(ticker, start, end, demo)
        df_ind = indicators.compute_all_indicators(df)
        signals = indicators.latest_signal_summary(df_ind)

        # 2. News + sentiment (qualitative layer)
        news_result = {"overall_sentiment_score": 0.0, "overall_summary": "Skipped.", "articles": []}
        if not skip_news:
            try:
                articles = news_fetch.fetch_company_news(company_name, ticker)
                news_result = sentiment.score_news_sentiment(company_name, ticker, articles)
            except Exception as e:
                news_result = {
                    "overall_sentiment_score": 0.0,
                    "overall_summary": f"News/sentiment step failed, continuing with quant-only prediction: {e}",
                    "articles": [],
                }

        # 3. Train models + predict
        feat_data = predictor.build_features(df_ind, horizon=1)
        if len(feat_data) < 60:
            raise HTTPException(
                status_code=422,
                detail=f"Only {len(feat_data)} usable rows of data — need at least ~100 for a meaningful model. "
                       f"Try a wider date range.",
            )

        dir_model, dir_metrics, feat_importance = predictor.train_direction_classifier(feat_data)
        price_model, price_metrics = predictor.train_price_regressor(feat_data)

        live_sentiment = news_result.get("overall_sentiment_score", 0.0)
        pred = predictor.predict_latest(dir_model, price_model, df_ind, live_sentiment_score=live_sentiment)

        return {
            "ticker": ticker,
            "company_name": company_name,
            "data_range": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date())},
            "n_trading_days": len(df),
            "technical_signals": signals,
            "news": news_result,
            "model_metrics": {
                "direction": dir_metrics,
                "price": price_metrics,
                "warning": (
                    "Model barely beats naive baseline — treat direction prediction with heavy skepticism."
                    if dir_metrics["accuracy"] - dir_metrics["baseline_accuracy"] < 0.03 else None
                ),
            },
            "feature_importance": feat_importance.round(4).to_dict(),
            "prediction": pred,
            "disclaimer": (
                "Educational/statistical tool, not financial advice. DSE stocks are "
                "thinly traded and news/policy-driven; no model captures every risk."
            ),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")