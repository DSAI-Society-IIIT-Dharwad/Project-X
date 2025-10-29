# 📰 Real-Time News Dashboard with Sentiment Analysis

A powerful news aggregation and analysis platform that combines Reddit and Google News with AI-powered sentiment analysis, topic modeling, and trend detection.

## 🚀 Features

- **📡 Multi-Source News Aggregation**
  - Reddit news from multiple subreddits
  - Google News with RSS feed integration
  - Custom search functionality for any topic

- **🤖 AI-Powered Analysis**
  - Real-time sentiment analysis (positive/negative/neutral)
  - Topic modeling with BERTopic
  - Trend detection and scoring
  - AI-generated summaries

- **📊 Interactive Dashboard**
  - Real-time news feeds side-by-side (Reddit vs Google News)
  - Sentiment distribution charts
  - Timeline visualization
  - Custom search with sentiment analysis

- **🔍 Advanced Features**
  - FAISS-based semantic search
  - Automatic duplicate detection
  - Refresh history tracking
  - Export capabilities

## 📁 File Structure

```
Project-X/
│
├── app/                          # Streamlit dashboard applications
│   ├── dashboard.py             # Main real-time news dashboard (Workflow 1)
│   ├── query_analyzer.py        # Query-based search interface (Workflow 2)
│   ├── chatbot.py               # Interactive chatbot interface
│   └── voice_interface.py       # Voice command interface
│
├── backend/                      # Backend API and database
│   ├── app.py                   # FastAPI backend server
│   ├── db.py                    # Database configuration and session management
│   ├── models.py                # SQLAlchemy database models
│   ├── schemas.py               # Pydantic schemas for API
│   ├── chat_storage.py          # Chat history and refresh tracking
│   └── ingest.py                # Data ingestion and processing utilities
│
├── pipeline/                     # NLP and ML processing pipeline
│   ├── embeddings.py            # Sentence embeddings generation
│   ├── sentiment.py             # Sentiment analysis using transformers
│   ├── topic_model.py           # Topic modeling with BERTopic
│   ├── trend_score.py           # Trend calculation and detection
│   ├── summary.py               # AI-generated summaries (Gemini)
│   ├── preprocess.py            # Text preprocessing utilities
│   └── quick_pipeline.py        # Quick processing for queries
│
├── scraper/                      # News scrapers
│   ├── reddit_scraper.py        # Reddit news scraper (PRAW)
│   ├── google_news_scraper.py   # Google News RSS scraper
│   ├── twitter_scraper.py       # Twitter scraper (if enabled)
│   └── push_shift_fallback.py   # PushShift API fallback
│
├── config.py                     # Configuration settings
├── launcher.py                   # System launcher with menu
├── quickstart.py                 # Quick start script
├── requirements.txt              # Python dependencies
└── README.md                     # This file

```

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (optional)

### Step 1: Clone or Navigate to Project

```bash
cd Project-X
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Reddit API Credentials (Required for Reddit scraping)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_name
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password

# Google Generative AI (Optional - for AI summaries)
GOOGLE_API_KEY=your_api_key

# Database (Optional - defaults to SQLite)
DATABASE_URL=sqlite+aiosqlite:///./news_bot.db
```

#### Getting Reddit API Credentials:

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..." or "create app"
3. Fill in:
   - Name: Your app name
   - Type: script
   - Description: optional
   - Redirect URI: http://localhost:8080
4. Note your client ID (under the app name) and secret
5. Add them to `.env` file

### Step 4: Initialize Database

```bash
python -c "import asyncio; from backend.db import init_db; asyncio.run(init_db())"
```

## 🎯 Quick Start

### Method 1: Using the Launcher (Recommended)

```bash
python launcher.py
```

Select option **1** to launch the dashboard.

### Method 2: Direct Launch

```bash
streamlit run app/dashboard.py
```

The dashboard will open at: `http://localhost:8501`

## 📖 Usage Guide

### Dashboard Overview

The dashboard has two main sections:

#### 1. Main Content Area
- **Metrics Bar**: Total posts, recent posts, avg sentiment, number of sources
- **Charts**: Sentiment distribution, source distribution, timeline
- **Recent Posts**: Side-by-side Reddit and Google News articles

#### 2. Sidebar
- **Refresh Button**: Scrapes new data from all sources
- **Custom Search**: Search any topic across Google News
- **Settings**: Time window selector (6, 12, 24, 48 hours)
- **Refresh History**: View past refresh operations
- **Cache Controls**: Clear cache when needed

### How to Use

1. **First Time Setup**
   - Click "Refresh Data" to scrape initial news
   - Wait for processing to complete
   - View results in the dashboard

2. **Viewing News**
   - Reddit posts show on the left
   - Google News articles show on the right
   - Click expandable items to see details

3. **Custom Search**
   - Type any topic in the sidebar search box
   - Click "Search News"
   - View results with sentiment analysis

4. **Generate Summaries**
   - After refreshing data, scroll down
   - Click "Generate Summary" button
   - View AI-generated trend summary

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Monitored subreddits
MONITORED_SUBREDDITS = ['news', 'worldnews', 'UpliftingNews', 'geopolitics']

# Google News settings
ENABLE_GOOGLE_NEWS = True
GOOGLE_NEWS_COUNTRY = "US"
GOOGLE_NEWS_LANGUAGE = "en"
GOOGLE_NEWS_QUERIES = ["breaking news", "world news", "technology news", "political news"]

# Scraping limits
POSTS_PER_SUBREDDIT = 50
GOOGLE_NEWS_LIMIT = 30

# ML features
ENABLE_EMBEDDINGS = True
ENABLE_SENTIMENT = True
ENABLE_TOPICS = True
ENABLE_SUMMARIES = True
```

## 🔄 Workflow

### Workflow 1: Real-Time Dashboard
1. User clicks "Refresh Data"
2. System scrapes Reddit and Google News
3. New posts saved to database
4. NLP pipeline processes posts (sentiment, embeddings, topics)
5. Dashboard updates with results

### Workflow 2: Query-Based Search
1. User enters custom search query
2. System searches Google News for that topic
3. Results displayed immediately
4. Sentiment analysis applied to results

## 🐛 Troubleshooting

### Issue: Google News not showing articles
**Solution**: 
- Click "Refresh Data" button in dashboard
- Wait for processing to complete
- Refresh the page

### Issue: Sentiment charts not displaying
**Solution**: 
- Ensure posts have been processed through the ML pipeline
- Click "Refresh Data" and wait for sentiment analysis to complete
- Check that `ENABLE_SENTIMENT = True` in config.py

### Issue: Summary generation failing
**Solution**: 
- Ensure you have Google API key in `.env` file
- Or disable summaries by setting `ENABLE_SUMMARIES = False`

### Issue: Reddit scraping not working
**Solution**: 
- Verify Reddit API credentials in `.env` file
- Check that Reddit subreddits in config are valid
- Ensure internet connection is active

### Issue: Dashboard not loading
**Solution**: 
```bash
# Kill existing Streamlit processes
ps aux | grep streamlit | grep -v grep | awk '{print $2}' | xargs kill -9

# Restart dashboard
streamlit run app/dashboard.py
```

## 🧪 Testing

### Test Scrapers Individually

```bash
# Test Reddit scraper
python -m scraper.reddit_scraper

# Test Google News scraper
python -m scraper.google_news_scraper

# Test query search
python -m scraper.reddit_scraper "artificial intelligence"
```

### Test Database

```bash
python -c "import asyncio; from backend.db import get_table_counts; print(asyncio.run(get_table_counts()))"
```

## 📊 API Endpoints (FastAPI Backend)

Start the API server:
```bash
uvicorn backend.app:app --reload --port 8000
```

Available at: `http://localhost:8000/docs`

### Key Endpoints:
- `GET /posts` - Get all posts
- `GET /trending` - Get trending posts
- `GET /topics` - Get all topics
- `GET /sentiment/stats` - Get sentiment statistics
- `POST /chat` - Chatbot interface

## 🎓 Key Technologies

- **Frontend**: Streamlit
- **Backend**: FastAPI, SQLAlchemy (async)
- **Database**: SQLite (default), PostgreSQL (production)
- **NLP**: Transformers, Sentence-BERT, BERTopic
- **ML**: PyTorch, scikit-learn, FAISS
- **Scraping**: PRAW (Reddit), feedparser (RSS)
- **Visualization**: Plotly, Matplotlib
- **AI Summaries**: Google Gemini API

## 📝 Dependencies

See `requirements.txt` for complete list. Key packages:

- streamlit - Dashboard
- fastapi - Backend API
- sqlalchemy - ORM
- transformers - NLP models
- sentence-transformers - Embeddings
- bertopic - Topic modeling
- torch - Deep learning
- praw - Reddit API
- feedparser - RSS parsing
- plotly - Charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Check logs in the terminal
4. Verify all dependencies are installed

## 🎯 Next Steps

After running the dashboard:

1. **Explore the Data**
   - Click through recent posts
   - View sentiment distributions
   - Analyze trends

2. **Customize**
   - Adjust subreddits in config.py
   - Modify Google News queries
   - Change scraping limits

3. **Extend Functionality**
   - Add new scrapers
   - Implement custom visualizations
   - Integrate additional ML models

## 📊 System Architecture

```
┌──────────────┐
│   Scrapers   │──────┐
│ (Reddit/GN)  │      │
└──────────────┘      │
                      ▼
              ┌──────────────┐
              │   Database   │
              │  (SQLite)    │
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │   Pipeline   │
              │ ens/sent/top │
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Dashboard   │
              │  (Streamlit) │
              └──────────────┘
```

## 🎉 Enjoy Your News Dashboard!

Happy news analyzing! 🚀📰

