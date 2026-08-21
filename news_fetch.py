"""
news_fetch.py
Fetches recent news headlines/articles relevant to a given DSE-listed company.

Strategy: RSS feeds are far more reliable than scraping raw HTML (sites
change layout, add anti-bot measures, etc). We pull from a few Bangladeshi
business news RSS feeds and Google News RSS (company name + "share" / "DSE"
as a search query), then filter for relevance to the ticker/company name.

If a feed breaks or a site blocks requests, this degrades gracefully:
returns an empty list rather than crashing the whole prediction pipeline.
"""

import feedparser
import requests
from datetime import datetime, timedelta
from urllib.parse import quote

# Bangladeshi business news RSS feeds (add more as you find reliable ones)
STATIC_FEEDS = [
    "https://www.thedailystar.net/business/rss.xml",
    "https://www.tbsnews.net/economy/stocks/rss.xml",
]

REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (compatible; DSE-Research-Tool/1.0)"


def _google_news_rss_url(query: str, days_back: int = 14) -> str:
    """Builds a Google News RSS search URL scoped to Bangladesh, recent news."""
    q = quote(f"{query} DSE OR share OR stock")
    return f"https://news.google.com/rss/search?q={q}&hl=en-BD&gl=BD&ceid=BD:en"


def fetch_company_news(company_name: str, ticker: str, days_back: int = 14, max_articles: int = 15) -> list[dict]:
    """
    Returns a list of {title, summary, link, published, source} dicts for
    recent news mentioning the company. Combines static feeds (filtered by
    keyword match) + a targeted Google News RSS search.
    """
    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []
    seen_titles = set()

    # 1. Targeted search feed (most relevant)
    try:
        search_url = _google_news_rss_url(company_name)
        feed = feedparser.parse(search_url)
        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            articles.append({
                "title": title,
                "summary": entry.get("summary", "")[:500],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Google News") if hasattr(entry.get("source", {}), "get") else "Google News",
            })
    except Exception as e:
        print(f"[news_fetch] Google News search failed: {e}")

    # 2. Static business feeds, filtered by keyword relevance
    keywords = [company_name.lower(), ticker.lower()]
    for feed_url in STATIC_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "")
                text = f"{title} {summary}".lower()
                if not any(kw in text for kw in keywords):
                    continue
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                articles.append({
                    "title": title,
                    "summary": summary[:500],
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url.split("/")[2],
                })
        except Exception as e:
            print(f"[news_fetch] Feed {feed_url} failed: {e}")

    return articles[:max_articles]


if __name__ == "__main__":
    results = fetch_company_news("Grameenphone", "GP")
    for a in results:
        print(f"- [{a['source']}] {a['title']}")
