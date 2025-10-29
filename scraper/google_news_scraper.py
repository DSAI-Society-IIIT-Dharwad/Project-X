"""
Google News Scraper
Fetches news articles from Google News with similar structure to Reddit scraper
"""

import feedparser
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
import time
import asyncio
import hashlib
from sqlalchemy import select

from backend.db import DatabaseSession, init_db
from backend.models import Post
from config import (
    MONITORED_SUBREDDITS, 
    POSTS_PER_SUBREDDIT,
    ENABLE_GOOGLE_NEWS,
    GOOGLE_NEWS_COUNTRY,
    GOOGLE_NEWS_LANGUAGE,
    GOOGLE_NEWS_QUERIES,
    GOOGLE_NEWS_LIMIT
)

load_dotenv()

# ============================================
# GOOGLE NEWS SCRAPER
# ============================================

def get_google_news_rss_url(query: Optional[str] = None, country: str = "US", language: str = "en"):
    """
    Generate Google News RSS feed URL
    
    Args:
        query: Search query (None for general news)
        country: Country code (US, UK, IN, etc.)
        language: Language code (en, es, etc.)
    
    Returns:
        RSS feed URL
    """
    base_url = "https://news.google.com/rss"
    
    if query:
        # Encode query for URL
        query_encoded = query.replace(' ', '+')
        return f"{base_url}/search?q={query_encoded}&hl={language}&gl={country}&ceid={country}:{language}"
    else:
        # Top headlines
        return f"{base_url}/headlines?hl={language}&gl={country}&ceid={country}:{language}"

def scrape_google_news(
    query: Optional[str] = None,
    limit: int = None,
    country: str = None,
    language: str = None
) -> List[Dict]:
    """
    Scrape news from Google News RSS feeds
    
    Args:
        query: Search query (None for general news)
        limit: Maximum number of articles
        country: Country code
        language: Language code
    
    Returns:
        List of article dictionaries
    """
    if limit is None:
        limit = GOOGLE_NEWS_LIMIT
    if country is None:
        country = GOOGLE_NEWS_COUNTRY
    if language is None:
        language = GOOGLE_NEWS_LANGUAGE
    
    print(f"\n📰 Scraping Google News")
    if query:
        print(f"   Query: '{query}'")
    print(f"   Limit: {limit}\n")
    
    all_articles = []
    
    try:
        # Get RSS feed URL
        rss_url = get_google_news_rss_url(query=query, country=country, language=language)
        
        print(f"📡 Fetching: {rss_url}")
        feed = feedparser.parse(rss_url)
        
        if feed.bozo:
            print(f"⚠️  Warning: Feed parsing issue - {feed.bozo_exception}")
        
        count = 0
        for entry in feed.entries[:limit]:
            try:
                # Extract article data
                title = entry.get('title', 'No title')
                link = entry.get('link', '')
                description = entry.get('description', entry.get('summary', ''))
                
                # Clean HTML tags from description
                import re
                if description:
                    description = re.sub('<[^<]+?>', '', description)  # Remove HTML tags
                    description = description.strip()
                
                # Parse published date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                else:
                    published_time = datetime.now(timezone.utc)
                
                # Get source
                if hasattr(entry, 'source') and entry.source:
                    source_name = entry.source.get('title', 'Unknown')
                else:
                    source_name = "Google News"
                
                # Create article data structure similar to Reddit
                article_data = {
                    "post_id": hashlib.md5(link.encode()).hexdigest(),  # Generate unique ID
                    "source": "google_news",
                    "subreddit": source_name,  # Use source name as equivalent to subreddit
                    "author": source_name,
                    "title": title,
                    "content": description,
                    "url": link,
                    "score": 0,  # Google News doesn't have upvotes
                    "num_comments": 0,  # Google News doesn't have comments
                    "created_at": published_time,
                    "permalink": link,
                    "workflow": "monitored"  # Mark source
                }
                
                all_articles.append(article_data)
                count += 1
            
            except Exception as e:
                print(f"❌ Error parsing entry: {e}")
                continue
        
        print(f"✅ Scraped {count} articles from Google News")
        
    except Exception as e:
        print(f"❌ Error scraping Google News: {e}")
    
    print(f"\n✅ Total: {len(all_articles)} articles from Google News")
    return all_articles

def scrape_monitored_google_news(
    queries: Optional[List[str]] = None,
    limit_per_query: int = None
) -> List[Dict]:
    """
    Scrape from monitored search queries
    
    Args:
        queries: List of search queries (defaults to monitored subreddits as queries)
        limit_per_query: Articles per query (defaults to config)
    
    Returns:
        List of article dictionaries
    """
    if limit_per_query is None:
        limit_per_query = POSTS_PER_SUBREDDIT
    
    # Default queries based on config
    if queries is None:
        queries = GOOGLE_NEWS_QUERIES
    
    print(f"\n📰 GOOGLE NEWS: Scraping monitored queries")
    print(f"   Queries: {queries}")
    print(f"   Articles per query: {limit_per_query}\n")
    
    all_articles = []
    
    for query in queries:
        try:
            articles = scrape_google_news(query=query, limit=limit_per_query)
            all_articles.extend(articles)
            time.sleep(2)  # Rate limiting
        except Exception as e:
            print(f"❌ Error for query '{query}': {e}")
    
    return all_articles

# ============================================
# DATA STORAGE
# ============================================

def save_to_json(articles: List[Dict], filename: Optional[str] = None, workflow: str = "monitored"):
    """Save scraped articles to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"google_news_{workflow}_{timestamp}.json"
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, default=str)
    
    print(f"💾 Saved to: {filepath}")
    return filepath

async def save_to_database(articles: List[Dict], workflow: str = "monitored"):
    """Save scraped articles to database"""
    print(f"\n💾 Saving {len(articles)} articles to database ({workflow})...")
    
    async with DatabaseSession() as db:
        saved_count = 0
        skipped_count = 0
        
        for article_data in articles:
            try:
                # Check if article already exists
                existing = await db.scalar(
                    select(Post).where(Post.post_id == article_data["post_id"])
                )
                
                if existing:
                    skipped_count += 1
                    continue
                
                # Create new post
                post = Post(
                    post_id=article_data["post_id"],
                    source=article_data["source"],
                    subreddit=article_data.get("subreddit"),
                    author=article_data["author"],
                    title=article_data["title"],
                    content=article_data.get("content", ""),
                    url=article_data.get("url"),
                    score=article_data["score"],
                    num_comments=article_data["num_comments"],
                    created_at=article_data["created_at"],
                    is_processed=False,
                    is_trending=False
                )
                
                db.add(post)
                saved_count += 1
            
            except Exception as e:
                print(f"❌ Error saving article {article_data.get('post_id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
        
        try:
            await db.commit()
        except Exception as e:
            print(f"❌ Error committing to database: {e}")
            await db.rollback()
    
    print(f"✅ Saved: {saved_count} | Skipped (duplicates): {skipped_count}")
    return {'saved': saved_count, 'skipped': skipped_count}

# ============================================
# MAIN FUNCTIONS
# ============================================

async def workflow_1_refresh():
    """
    Workflow 1: Refresh Google News
    Called when user clicks "Refresh" on dashboard
    """
    if not ENABLE_GOOGLE_NEWS:
        return {'success': True, 'posts_scraped': 0, 'posts_saved': 0, 'posts_skipped': 0}
    
    print("="*60)
    print("🔄 WORKFLOW 1: Refreshing Google News")
    print("="*60)
    
    await init_db()
    
    # Scrape Google News
    articles = scrape_monitored_google_news()
    
    if articles:
        # Save to JSON backup
        save_to_json(articles, workflow="monitored")
        
        # Save to database
        stats = await save_to_database(articles, workflow="monitored")
        
        return {
            'success': True,
            'posts_scraped': len(articles),
            'posts_saved': stats['saved'],
            'posts_skipped': stats['skipped']
        }
    
    return {'success': False, 'error': 'No articles scraped'}

# ============================================
# CLI ENTRY POINTS
# ============================================

async def main():
    """Default: Run Workflow 1"""
    result = await workflow_1_refresh()
    print("\n" + "="*60)
    print(f"✅ Google News Workflow Complete: {result}")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line query
        query = ' '.join(sys.argv[1:])
        articles = scrape_google_news(query=query)
        print(f"\n📊 Found {len(articles)} articles")
    else:
        # Default: monitored queries
        asyncio.run(main())

