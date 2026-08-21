# DSE Signal — AI Stock Terminal

A full web app for the Dhaka Stock Exchange: technical indicators + ML price
prediction (quantitative) fused with Claude-scored news sentiment
(qualitative), served through a FastAPI backend and a single-page dashboard.

```
frontend/index.html   — dashboard (vanilla JS, Chart.js via CDN, no build step)
backend/
  app.py               — FastAPI routes
  data_fetch.py         — DSE price data (bdshare) + synthetic demo data
  news_fetch.py          — news headlines (RSS: Google News + BD business feeds)
  sentiment.py            — Claude API scores news for sentiment/relevance
  indicators.py             — SMA/EMA/RSI/MACD/Bollinger/volatility
  predictor.py                — Random Forest models (direction + price), fused with sentiment
  requirements.txt
render.yaml            — one-click Render deployment config
```

## How it fits together

1. You search a ticker (e.g. `GP`) on the dashboard.
2. Backend fetches historical price data → computes technical indicators.
3. Backend fetches recent news about the company → scores each article with
   a **locally-run DistilBERT sentiment model** (no external AI API, no API
   key, no per-request cost) layered with a small financial-keyword
   adjustment for domain awareness.
4. A Random Forest model (trained fresh on that ticker's price history each
   request) predicts next-day direction and price, using both the technical
   indicators AND today's live sentiment score as inputs.
5. Dashboard renders the prediction, technical signals, price chart, and
   the scored news feed.

## Local setup (test before deploying)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

No API key needed anymore — the sentiment model runs inside your own
process. First request that touches `/api/predict/...` with news enabled
will take longer (downloading + loading the ~265MB model), subsequent
requests reuse the cached model.

Open `frontend/index.html` directly in your browser (no server needed for
the frontend itself — it's a static file). It's already pointed at
`http://localhost:8000`. Check "Demo mode" to test with synthetic data
first.

## Deploying for real (so it works from your phone, anywhere)

### Backend → Render

1. Push this whole folder to a GitHub repo.
2. Go to [render.com](https://render.com) → New → Blueprint → connect your repo.
   Render will read `render.yaml` and set everything up automatically.
   (Or manually: New → Web Service → root directory `backend` → build
   command `pip install -r requirements.txt` → start command
   `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`.)
3. No API key to configure — the sentiment model is self-hosted.
4. Deploy. You'll get a URL like `https://dse-ai-predictor.onrender.com`.

**Free tier note:** Render's free web services spin down after 15 minutes
of inactivity and take ~30-60 seconds to wake up on the next request. The
sentiment model reloads on the first request after each wake-up (not
persisted across spin-downs on the free tier), adding a few extra seconds
to that first call.

**Memory note (important):** the sentiment model (~265MB) plus FastAPI,
pandas, scikit-learn, and bdshare can be tight on Render's free 512MB
instance. If the service crashes or restarts under load:
- Keep `--workers 1` in the start command (already set in `render.yaml`)
  — multiple workers would each load their own copy of the model and blow
  past 512MB immediately.
- Try `?skip_news=true` on the predict endpoint to confirm the quant-only
  path works fine even if the sentiment path is OOM-ing.
- If it's still too tight, the next step down in size is an
  **ONNX-quantized** version of the same model (~65-100MB instead of
  ~265MB) via the `optimum[onnxruntime]` package — more setup work, worth
  it only if the free tier truly won't hold the current model.
- Otherwise, Render's paid Starter/Standard tiers give more headroom.

### Frontend → Netlify

1. Open `frontend/index.html`, change this line near the top of the
   `<script>` block:
   ```js
   const API_BASE = "http://localhost:8000";
   ```
   to your Render URL:
   ```js
   const API_BASE = "https://dse-ai-predictor.onrender.com";
   ```
2. Drag-and-drop the `frontend` folder into Netlify (same workflow you
   already use for other projects) — no build step needed, it's one static
   HTML file.
3. (Optional, more secure) In `backend/app.py`, tighten the CORS
   `allow_origins=["*"]` to your actual Netlify URL once you know it.

## Honest limitations — read before trusting predictions

- **Sentiment is "live only," not historical.** The model trains on price
  history where sentiment is treated as neutral (0) for every past day,
  because scoring years of historical news for every ticker isn't
  practical here. At prediction time, today's real sentiment score gets
  injected. This means sentiment mostly nudges today's specific prediction
  rather than being something the model has learned deep patterns from.
  To fix this properly: build a news archive over time (store each day's
  score to a small database) and retrain periodically — the code is
  structured so this is a straightforward extension, not a rewrite.
- **Sentiment model is general-purpose, not finance-tuned, and English-only.**
  DistilBERT-SST2 was trained on movie reviews, not financial news — it's
  good at picking up positive/negative tone but won't understand finance-
  specific nuance the way a model trained on financial text would. The
  keyword layer (`POSITIVE_TERMS`/`NEGATIVE_TERMS` in `sentiment.py`)
  patches some of this gap, but it's a heuristic, not learned understanding.
  It also only reads English — Bangla-language headlines (e.g. Prothom Alo)
  won't be scored correctly unless you swap in a multilingual model (see
  "Extending this later").
- **DSE has no official API.** `bdshare` scrapes dsebd.org; if DSE changes
  their site structure, `data_fetch.py` will need updating.
- **News sources are RSS-based**, which is more reliable than raw scraping
  but still depends on those feeds staying live and relevant. Bangladeshi
  company-specific financial news is sparser than for US/global stocks —
  some tickers may return zero articles, and the model will fall back to
  quant-only prediction (this is handled gracefully, not a crash).
- **Every model retrains from scratch on each request** (no model
  persistence/caching yet). Fine for individual tickers; if you want this
  to scale to many concurrent users, add a cache layer (e.g. retrain once
  per ticker per day, store the model, serve cached predictions).
- **Free-tier memory is tight.** See the Render deployment section above —
  the sentiment model plus everything else may need `--workers 1` and
  careful monitoring on a 512MB instance.
- **This is not financial advice.** DSE stocks are thin and news/policy-
  driven; no model — including this one — captures every risk. The
  dashboard shows model accuracy against a naive baseline for exactly this
  reason: so you can see when the model isn't actually adding value for a
  given stock, rather than trusting every prediction equally.

## Extending this later

- **Bangla support**: swap `MODEL_NAME` in `sentiment.py` for a multilingual
  model like `cardiffnlp/twitter-xlm-roberta-base-sentiment` (bigger, ~1.1GB
  — needs a paid hosting tier) so Bangla-language headlines score correctly
  too, not just English ones
- **Smaller footprint**: convert the model to ONNX + int8 quantization via
  `optimum[onnxruntime]` for a ~65-100MB footprint instead of ~265MB, if
  free-tier memory stays tight
- Cache/persist trained models per ticker (Redis or a simple SQLite table)
  instead of retraining on every request
- Build a real sentiment history by storing daily scores, enabling proper
  backtesting of the sentiment feature (not just live nudging)
- Add sector-relative indicators (compare a stock to its DSE sector index)
- Multi-ticker comparison view
- Swap Random Forest for LSTM once you have a persistent, larger dataset
