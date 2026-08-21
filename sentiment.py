"""
sentiment.py
Local, self-hosted sentiment scoring — NO external AI API (no Claude, no
OpenAI, nothing that costs per-call or needs an API key). This is a real
neural network (DistilBERT) that runs inside your own backend process.

Design choices, and why:

1. MODEL: distilbert-base-uncased-finetuned-sst-2-english
   - ~66M parameters, ~265MB on disk. This is a *distilled* model (a
     compressed version of BERT that keeps ~97% of its language
     understanding at ~40% of the size), specifically so it can run on a
     small/free hosting tier without a GPU.
   - It's a GENERAL sentiment model (positive/negative), not finance-tuned.
     Full finance-tuned models (FinBERT etc.) are BERT-base sized (~440MB)
     and multilingual models are bigger still (~500MB-1GB+) — too heavy for
     a free-tier instance alongside FastAPI/pandas/sklearn.
   - To compensate for it not being finance-specific, we layer a small
     FINANCIAL KEYWORD ADJUSTMENT on top (see below). This is a real,
     honest trade-off: less nuanced than a true finance-tuned model, but
     free, fast, and self-hosted.

2. LAZY LOADING: the model loads on the FIRST request that needs it, not at
   server startup. This keeps health checks and startup fast, and means
   you're not holding model weights in memory if nobody's using the
   sentiment feature yet.

3. LANGUAGE: this build targets English-language news (our news_fetch.py
   already pulls from English-language BD business feeds + an English
   Google News query). If you want Bangla headlines scored too, see the
   "Extending to Bangla" note in README.md.

MEMORY NOTE FOR FREE-TIER HOSTING: even at ~265MB, this model plus
FastAPI/pandas/scikit-learn may be tight on a 512MB instance. If you hit
out-of-memory errors on Render's free tier, either (a) upgrade to the
$7/mo Starter plan (512MB -> still tight, prefer Standard 2GB), or
(b) see the ONNX-quantization note in README.md for a ~4x smaller model.
"""

import re
from functools import lru_cache

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# --- Financial domain keyword layer -----------------------------------
# A lightweight, transparent nudge on top of the neural model's raw
# positive/negative read, so obviously finance-relevant language isn't
# missed just because the base model wasn't finance-tuned.
POSITIVE_TERMS = {
    "profit": 0.15, "profit growth": 0.25, "record profit": 0.3, "surge": 0.2,
    "dividend": 0.15, "bonus share": 0.15, "rally": 0.2, "outperform": 0.2,
    "expansion": 0.1, "revenue growth": 0.2, "upgrade": 0.15, "beat estimates": 0.25,
    "strong earnings": 0.25, "capital gain": 0.15, "buyback": 0.15,
}
NEGATIVE_TERMS = {
    "loss": -0.2, "net loss": -0.3, "fine": -0.2, "penalty": -0.25,
    "regulatory action": -0.2, "circuit breaker": -0.2, "fraud": -0.4,
    "investigation": -0.25, "scandal": -0.35, "downgrade": -0.2,
    "default": -0.3, "delisting": -0.35, "suspension": -0.25,
    "profit warning": -0.3, "shortfall": -0.2, "layoffs": -0.15,
}


def _keyword_adjustment(text: str) -> float:
    text_lower = text.lower()
    score = 0.0
    for term, weight in POSITIVE_TERMS.items():
        if term in text_lower:
            score += weight
    for term, weight in NEGATIVE_TERMS.items():
        if term in text_lower:
            score += weight
    return max(-0.5, min(0.5, score))  # cap the keyword layer's influence


@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Loads the model on first call, caches it for the life of the process.
    Using lru_cache instead of a module-level global keeps this explicit
    and testable.
    """
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)  # device=-1 = CPU


def _score_single(pipe, text: str) -> tuple[float, str]:
    """
    Returns (signed_score, reason). The base model gives a label + a
    confidence in [0,1]; we convert to a signed score in [-1, 1], then
    blend in the keyword adjustment.
    """
    text = text[:512]  # DistilBERT's context window; longer text gets truncated
    result = pipe(text)[0]
    label = result["label"]  # "POSITIVE" or "NEGATIVE"
    conf = result["score"]

    base_score = conf if label == "POSITIVE" else -conf
    kw_adj = _keyword_adjustment(text)
    final_score = max(-1.0, min(1.0, base_score * 0.7 + kw_adj))

    reason_bits = [f"model read: {label.lower()} ({conf:.2f} confidence)"]
    if abs(kw_adj) > 0.01:
        reason_bits.append(f"finance-keyword adjustment: {kw_adj:+.2f}")
    return final_score, "; ".join(reason_bits)


def _relevance_heuristic(text: str, company_name: str, ticker: str) -> str:
    text_lower = text.lower()
    hits = sum(1 for kw in [company_name.lower(), ticker.lower()] if kw in text_lower)
    if hits >= 2 or company_name.lower() in text_lower:
        return "high"
    elif hits == 1:
        return "medium"
    return "low"


def score_news_sentiment(company_name: str, ticker: str, articles: list[dict]) -> dict:
    """
    Same return contract as the old Claude-based version, so app.py and
    the frontend don't need any changes:
      {overall_sentiment_score, overall_summary, articles: [...]}
    """
    if not articles:
        return {
            "overall_sentiment_score": 0.0,
            "overall_summary": "No recent news found for this company.",
            "articles": [],
        }

    pipe = _get_pipeline()

    scored = []
    weighted_sum = 0.0
    weight_total = 0.0
    relevance_weight = {"high": 1.0, "medium": 0.5, "low": 0.15}

    for a in articles:
        text = f"{a.get('title', '')}. {a.get('summary', '')}"
        score, reason = _score_single(pipe, text)
        relevance = _relevance_heuristic(text, company_name, ticker)

        scored.append({
            "title": a.get("title", ""),
            "link": a.get("link", ""),
            "source": a.get("source", ""),
            "sentiment_score": round(score, 3),
            "relevance": relevance,
            "reason": reason,
        })

        w = relevance_weight[relevance]
        weighted_sum += score * w
        weight_total += w

    overall = weighted_sum / weight_total if weight_total > 0 else 0.0

    if overall > 0.15:
        tone = "net positive"
    elif overall < -0.15:
        tone = "net negative"
    else:
        tone = "mixed/neutral"

    summary = (
        f"{len(articles)} article(s) scored locally (DistilBERT + financial "
        f"keyword layer). Overall tone: {tone} ({overall:+.2f})."
    )

    return {
        "overall_sentiment_score": round(overall, 3),
        "overall_summary": summary,
        "articles": scored,
    }


if __name__ == "__main__":
    sample_articles = [
        {"title": "Grameenphone reports record profit growth in Q2", "summary": "Strong subscriber growth drove earnings.", "source": "TBS News", "link": ""},
        {"title": "BTRC fines Grameenphone over service quality investigation", "summary": "Regulator cites call drop rates above threshold.", "source": "Daily Star", "link": ""},
    ]
    import json
    print(json.dumps(score_news_sentiment("Grameenphone", "GP", sample_articles), indent=2))
