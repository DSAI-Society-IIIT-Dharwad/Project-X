📰 NewsBot: Public Sentiment vs Media Analytics
Table of Contents
Project Overview

Demo Screenshot

Architecture

How It Works

Features

Quick Start Guide

Core Workflows

Key Code Explanations

File Structure

Contributing

License

Project Overview
NewsBot is a real-time analytics platform for comparing public opinion (Reddit) and mainstream media (Google News) — discover sentiment, trending news, and topic divergence instantly.

Dynamic filters for selecting relevant subreddits and news sources

Automated sentiment analysis and interactive comparison

Live previews of top posts, articles, and trending keywords

Built-in AI chat assistant for news Q&A

Modern interactive dashboard for clear visualization

Demo Screenshot
(Add your Streamlit dashboard screenshot here for the judges!)

Architecture
text
graph TD
    A[User Filter/Query] --> B[Streamlit Dashboard]
    B --> C[Scraper Layer]
    C --> D[(Database)]
    D --> E[Processing Pipeline]
    E --> F[Sentiment & Topic Models]
    F --> B
    B --> G[AI Chat Assistant]
Dashboard: User interface, filters, charts, and AI chat

Scrapers: Targeted Reddit & Google News collection

Database: SQLite for posts and analyses

Pipeline: Sentiment models, embeddings, topic extraction

AI Chat: News Q&A with summarization

How It Works
Select sources: Choose subreddits and news outlets in the sidebar.

Trigger Refresh: Scrapes only the selected sources for fast, relevant data.

Processing Pipeline: Applies sentiment models, extracts keywords, detects topics.

Visualization: Interactive dashboard with charts, metrics, live previews.

Comparative Analysis: Judges can instantly see how Reddit and news media differ or agree.

Features
Side-by-side comparison: Reddit vs Google News for any selection

Multi-select filtering: Always starts empty—only analyzes user’s chosen sources

Preview cards: For each selected item; shows post counts, top post title, top keywords

Topic Comparison: Real-time query for any news topic across both sources

AI-powered assistant: Ask anything, get summarized news answers from current data

Stunning plots: Gauge charts, pie charts, and smart metric cards

Quick Start Guide
shell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize the database
python -c "import asyncio; from backend.db import init_db; asyncio.run(init_db())"

# 3. (Optional) Scrape initial data
python -m scraper.reddit_scraper
python -m scraper.google_news_scraper

# 4. Launch dashboard
streamlit run app/dashboard.py
Core Workflows
Filtered Scraping
python
# Scrape only selected subreddits (called from dashboard)
await workflow_1_refresh(subreddits=['news', 'worldnews', 'technology'])

# Scrape only selected Google News topics
await workflow_google_news(topics=['artificial intelligence', 'cryptocurrency'])
Dashboard Filtering & Preview
Select sources via sidebar multi-selects

For each subreddit or news site:

View post/article count

Recent (24h) count

Average sentiment

Top post/article preview

Top 5 keywords (trending topics)

Sentiment Comparison
Enter topic (e.g. "climate change")

Instantly compare sentiment and coverage across Reddit and Google News

Visual gauge and pie charts for clarity

AI Chat Assistant
Ask any news question

Assistant finds, analyzes, and summarizes answers using the latest data

Provides supporting examples, confidence scores, and source links

Key Code Explanations
Scraper Dynamic Filtering
python
def scrape_monitored_subreddits(reddit, subreddits=None, limit=None):
    if subreddits is None:
        # Uses dashboard selection or fallback defaults.
        subreddits = MONITORED_SUBREDDITS
    ...
    for subreddit_name in subreddits:
        # Only scrapes what the user selects.
Dashboard Source Preview
python
def get_source_preview(source_name, source_type='reddit'):
    ...
    # Extract top keywords from recent titles.
    words = [w.lower() for title in titles for w in title.split() if len(w) > 4]
    top_keywords = [word for word, count in Counter(words).most_common(5)]
Filter UI in Dashboard
python
selected_reddit = st.multiselect(
    "Select subreddits:",
    options=filtered_reddit,
    default=[],
    key="reddit_filter"
)
File Structure
text
news_bot/
├── app/
│   ├── dashboard.py            # Main interactive dashboard
│   ├── ai_assistant.py         # News chat assistant
├── backend/
│   ├── db.py                   # Database/SQLAlchemy setup
│   ├── models.py               # Data models
│   ├── ingest.py               # Processing pipeline
├── scraper/
│   ├── reddit_scraper.py       # Reddit data collector (now supports filters!)
│   ├── google_news_scraper.py  # Google News collector (supports filters!)
├── pipeline/
│   ├── sentiment.py            # Sentiment analysis (AI, transformers)
│   ├── embeddings.py           # Semantic similarity search (FAISS)
│   ├── topic_model.py          # BERTopic clustering
├── config.py                   # (Optional) default sources, settings
├── requirements.txt            # Python dependencies
└── README.md                   # (This file)
Contributing
Fork, create feature branch, submit PR.

Code must be clear, modular, and demo-friendly!

Filter-first logic, analytics-focused design.

License
MIT License – open for research, competition, and academic fair use.

Credits / Inspiration
PRAW, gnews, Streamlit, SQLAlchemy, HuggingFace Transformers, Gemini, BERTopic, FAISS.

Thanks to open APIs and developer communities.

