# 📰 Inquiro: Real-Time News Sentiment Analytics  
### _Unifying social signals and media coverage for deep, actionable insight._

---

## 🚀 Vision and Purpose

**NewsBot** revolutionizes news analytics by bridging the gap between **public sentiment** and **mainstream media reporting**.  
Rather than relying on either headlines **or** social chatter, NewsBot empowers users, analysts, and judges to see—**with one click**—whether Reddit’s public opinion aligns or diverges from Google News coverage on any topic, region, or event.

### It answers key questions:
- Is the public more optimistic, skeptical, or hostile than traditional media?
- What topics are trending organically on Reddit vs covered by the media?
- How does sentiment evolve in real-time across both sources?

This system goes beyond dashboards — it enables:
- **Rumor and crisis detection**
- **PR and media analysis**
- **Transparent, unbiased journalism**
- **Strategic decision-making through real-time sentiment data**

---

## 🧠 System Overview

### 🧩 1. Streamlit Dashboard (User Interface)
- A **modern, tabbed, interactive dashboard** built with Streamlit.  
- Sidebar contains **dynamic, searchable multi-selects** for:
  - Reddit subreddits  
  - Google News publishers/topics  
- Users start with **zero defaults** (no bias, full control).
- As sources are selected:
  - Preview cards show:
    - Total posts
    - 24h new content
    - Average sentiment
    - Top post/article
    - Trending keywords

---

### 🕸️ 2. Scraping Layer (Targeted & Efficient)
- **Reddit Scraper (`scraper/reddit_scraper.py`)**  
  Uses **PRAW** to scrape *only* selected subreddits for high speed.
- **Google News Scraper (`scraper/google_news_scraper.py`)**  
  Uses the **gnews API** to fetch filtered news from selected publishers/topics.

🧩 Fully dynamic — scrapes **only what the user selects**, saving time and bandwidth.

---

### 💾 3. Data Storage
- **SQLite database** for lightweight and fast local storage.
- Separate tables for Reddit and Google News posts for clarity.
- Stored fields include:
  - Timestamps
  - Source
  - Title/content
  - Sentiment
  - Metadata & links

---

### 🤖 4. Processing & NLP Pipeline
All natural language processing tasks are handled by modular transformer-based pipelines:

| Task | Description | Technology |
|------|--------------|-------------|
| **Sentiment Analysis** | Assigns sentiment scores from -1 (negative) → +1 (positive) | Transformer (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| **Keyword Extraction** | Highlights trending terms & entities | Frequency analysis |
| **Embeddings** | Vectorizes text for semantic similarity & clustering | Sentence-Transformers + FAISS |
| **Topic Modeling** | Groups related content for drift or trend detection | BERTopic |

All steps are orchestrated via `backend/ingest.py`.

---

### 🧭 5. AI Assistant
A real-time **AI-powered Q&A assistant** built into the dashboard.

Users can type queries like:  
> “What are the public concerns about AI this week?”

The assistant:
- Searches both Reddit & Google News
- Summarizes recent posts/articles
- Returns a concise, **source-cited**, sentiment-aware summary  
*(powered by Gemini or similar LLM integration)*

---

### 📊 6. Visualization
Beautiful, intuitive, real-time analytics powered by Streamlit and Plotly:

- **Gauge & Pie Charts** → Compare Reddit vs News sentiment
- **Metric Cards** → Post counts, trends, alignment score
- **Keyword Clouds** → Trending terms at a glance
- **Post Lists** → Dive deep into most recent or impactful posts

One refresh updates all data in real-time.

---

### ⚖️ 7. Comparison & Reporting
Judges or users can enter **any topic** (e.g., *“climate change”*, *“AI regulation”*, *“Ukraine”*)  
to instantly compare Reddit vs News sentiment.

✅ Highlights:
- **Aligned Sentiments** → "Sentiments ALIGN"  
- **Diverging Opinions** → "Sentiments DIFFER"

Perfect for transparency, public opinion monitoring, and trend evaluation.

---

## 💡 Features That Stand Out

| Category | Highlights |
|-----------|-------------|
| **Dynamic Architecture** | Nothing analyzed by default — user selects what matters |
| **Live Previews** | See top posts & stats before running full analysis |
| **Dual-Sourcing** | True side-by-side comparison: Reddit vs News |
| **AI Chat Assistant** | Instant, source-backed summaries & Q&A |
| **Open & Extensible** | SQLite + Open APIs + modular backend |
| **Transparency** | All pipelines open-source and explainable |
| **Performance** | Lightweight, targeted scraping; fast updates |

---

## 🧱 High-Level Module Structure


PRAW, gnews, Streamlit, SQLAlchemy, HuggingFace Transformers, Gemini, BERTopic, FAISS.

Thanks to open APIs and developer communities.




