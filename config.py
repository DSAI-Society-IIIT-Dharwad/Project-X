"""
Configuration for NewsBot
"""

# Workflow 1: Monitored subreddits for real-time dashboard
MONITORED_SUBREDDITS = [
    'news',
    'worldnews', 
    'UpliftingNews',
    'geopolitics'
]

# Google News settings
ENABLE_GOOGLE_NEWS = True  # Enable/disable Google News scraping
GOOGLE_NEWS_COUNTRY = "US"  # Country code for Google News
GOOGLE_NEWS_LANGUAGE = "en"  # Language code for Google News
GOOGLE_NEWS_QUERIES = [
    "breaking news",
    "world news",
    "technology news",
    "political news"
]

# Scraping settings
POSTS_PER_SUBREDDIT = 50  # How many posts to fetch per subreddit
GOOGLE_NEWS_LIMIT = 30  # How many articles to fetch from Google News
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
