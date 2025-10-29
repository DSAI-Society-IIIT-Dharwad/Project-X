"""
Configuration for AI News Assistant
Loads settings from environment variables with sensible defaults
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# API CONFIGURATION
# ============================================

# Reddit API Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "hackathon-news-bot:1.0")

# Google Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============================================
# AI MODEL CONFIGURATION
# ============================================

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOPIC_MODEL_NAME = os.getenv("TOPIC_MODEL_NAME", "bertopic_model")

# Vector Database Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./models/faiss_index/")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ============================================
# DATABASE CONFIGURATION
# ============================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./news_bot.db")

# ============================================
# SCRAPING CONFIGURATION
# ============================================

SCRAPE_LIMIT = int(os.getenv("SCRAPE_LIMIT", "100"))

# Monitored subreddits for real-time dashboard
MONITORED_SUBREDDITS = os.getenv("SUBREDDITS", "news,worldnews,UpliftingNews,nottheonion,politics,business,finance,economics,science,technology,health,environment,space").split(",")

# Keywords for trend detection
TREND_KEYWORDS = os.getenv("KEYWORDS", "breaking,trending,urgent,developing,announced,reports,confirms,major").split(",")

# Scraping settings
POSTS_PER_SUBREDDIT = 50  # How many posts to fetch per subreddit
REFRESH_INTERVAL_MINUTES = 30  # How often dashboard auto-refreshes

# Workflow 2: Query-based search settings
QUERY_SEARCH_LIMIT = 30  # Posts to fetch per query
QUERY_TIME_FILTER = 'week'  # 'hour', 'day', 'week', 'month'

# Processing settings
ENABLE_EMBEDDINGS = True
ENABLE_SENTIMENT = True
ENABLE_TOPICS = True
ENABLE_SUMMARIES = True

# Database settings
USE_SEPARATE_QUERY_DB = False  # If True, query results don't mix with main data

SEARCH_ALL_REDDIT = True  # Search entire Reddit, not just monitored subs
EXCLUDE_NSFW = True  # Skip NSFW content

# Google News Configuration
GOOGLE_NEWS_TOPICS = [
    "artificial intelligence",
    "climate change",
    "space exploration",
    "cryptocurrency",
    "politics",
    "technology"
]

GOOGLE_NEWS_PERIOD = '7d'  # '1h', '1d', '7d', '1m', '1y'
GOOGLE_NEWS_MAX_RESULTS = 50

EXCLUDED_SUBREDDITS = [
    'circlejerk',
    'copypasta',
    'shitposting'
]

# ============================================
# VOICE CONFIGURATION
# ============================================

VOICE_ENABLED = os.getenv("VOICE_ENABLED", "True").lower() == "true"
VOICE_LANGUAGE = os.getenv("VOICE_LANGUAGE", "en")

# ============================================
# SERVER CONFIGURATION
# ============================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "1000"))

# ============================================
# DEVELOPMENT CONFIGURATION
# ============================================

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================
# VALIDATION
# ============================================

def validate_config():
    """Validate required configuration"""
    required_vars = [
        ("REDDIT_CLIENT_ID", REDDIT_CLIENT_ID),
        ("REDDIT_CLIENT_SECRET", REDDIT_CLIENT_SECRET),
        ("GEMINI_API_KEY", GEMINI_API_KEY),
    ]
    
    missing_vars = []
    for var_name, var_value in required_vars:
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        return False
    
    print("✅ Configuration validated successfully!")
    return True

if __name__ == "__main__":
    validate_config()