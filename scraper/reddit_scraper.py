"""
Reddit Scraper using PRAW
Collects posts and comments from specified subreddits
"""

import praw
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
import time
import asyncio
from sqlalchemy import select

from backend.db import DatabaseSession, init_db
from backend.models import Post

# Load environment variables
load_dotenv()

# ============================================
# REDDIT CLIENT SETUP
# ============================================

def get_reddit_client():
    """Initialize and return Reddit client"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
        # Test authentication
        print(f"✅ Authenticated as: {reddit.user.me()}")
        return reddit
    
    except Exception as e:
        print(f"❌ Reddit authentication failed: {e}")
        print("\n🔑 Make sure your .env file has:")
        print("   REDDIT_CLIENT_ID")
        print("   REDDIT_CLIENT_SECRET")
        print("   REDDIT_USER_AGENT")
        raise


# ============================================
# SCRAPING FUNCTIONS
# ============================================

def scrape_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    limit: int = 100,
    time_filter: str = "day",
    sort_by: str = "hot"
) -> List[Dict]:
    """
    Scrape posts from a subreddit
    
    Args:
        reddit: PRAW Reddit instance
        subreddit_name: Name of subreddit (e.g., 'news')
        limit: Number of posts to fetch
        time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
        sort_by: 'hot', 'new', 'top', 'rising'
    
    Returns:
        List of post dictionaries
    """
    print(f"\n📡 Scraping r/{subreddit_name} ({sort_by}, limit={limit})...")
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts_data = []
        
        # Choose sorting method
        if sort_by == "hot":
            posts = subreddit.hot(limit=limit)
        elif sort_by == "new":
            posts = subreddit.new(limit=limit)
        elif sort_by == "top":
            posts = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort_by == "rising":
            posts = subreddit.rising(limit=limit)
        else:
            posts = subreddit.hot(limit=limit)
        
        for submission in posts:
            # Skip stickied posts
            if submission.stickied:
                continue
            
            post_data = {
                "post_id": submission.id,
                "source": "reddit",
                "subreddit": subreddit_name,
                "author": str(submission.author) if submission.author else "[deleted]",
                "title": submission.title,
                "content": submission.selftext if submission.selftext else "",
                "url": submission.url,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_at": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                "permalink": f"https://reddit.com{submission.permalink}",
                
                # Metadata for later processing
                "upvote_ratio": submission.upvote_ratio,
                "is_original_content": submission.is_original_content,
                "is_video": submission.is_video,
                "link_flair_text": submission.link_flair_text,
            }
            
            posts_data.append(post_data)
        
        print(f"✅ Scraped {len(posts_data)} posts from r/{subreddit_name}")
        return posts_data
    
    except Exception as e:
        print(f"❌ Error scraping r/{subreddit_name}: {e}")
        return []


def scrape_multiple_subreddits(
    reddit: praw.Reddit,
    subreddits: List[str],
    limit_per_sub: int = 100,
    time_filter: str = "day",
    sort_by: str = "hot"
) -> List[Dict]:
    """
    Scrape posts from multiple subreddits
    
    Args:
        reddit: PRAW Reddit instance
        subreddits: List of subreddit names
        limit_per_sub: Posts to fetch per subreddit
        time_filter: Time filter for sorting
        sort_by: Sorting method
    
    Returns:
        Combined list of all posts
    """
    all_posts = []
    
    for subreddit in subreddits:
        posts = scrape_subreddit(
            reddit=reddit,
            subreddit_name=subreddit,
            limit=limit_per_sub,
            time_filter=time_filter,
            sort_by=sort_by
        )
        all_posts.extend(posts)
        
        # Rate limiting - be nice to Reddit API
        time.sleep(2)
    
    print(f"\n✅ Total posts scraped: {len(all_posts)}")
    return all_posts


def scrape_with_keywords(
    reddit: praw.Reddit,
    subreddit_name: str,
    keywords: List[str],
    limit: int = 100
) -> List[Dict]:
    """
    Search for posts containing specific keywords
    
    Args:
        reddit: PRAW Reddit instance
        subreddit_name: Subreddit to search in
        keywords: List of keywords to search for
        limit: Maximum posts to return
    
    Returns:
        List of matching posts
    """
    print(f"\n🔍 Searching r/{subreddit_name} for keywords: {keywords}")
    
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    for keyword in keywords:
        try:
            search_results = subreddit.search(
                query=keyword,
                limit=limit // len(keywords),
                time_filter="day",
                sort="relevance"
            )
            
            for submission in search_results:
                post_data = {
                    "post_id": submission.id,
                    "source": "reddit",
                    "subreddit": subreddit_name,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "title": submission.title,
                    "content": submission.selftext,
                    "url": submission.url,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_at": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    "matched_keyword": keyword
                }
                posts_data.append(post_data)
            
            time.sleep(1)  # Rate limiting
        
        except Exception as e:
            print(f"❌ Error searching for '{keyword}': {e}")
    
    print(f"✅ Found {len(posts_data)} posts matching keywords")
    return posts_data


# ============================================
# DATA STORAGE
# ============================================

def save_to_json(posts: List[Dict], filename: Optional[str] = None):
    """Save scraped posts to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_scrape_{timestamp}.json"
    
    # Ensure data/raw directory exists
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, default=str)
    
    print(f"💾 Saved to: {filepath}")
    return filepath


async def save_to_database(posts: List[Dict]):
    """Save scraped posts to database"""
    print(f"\n💾 Saving {len(posts)} posts to database...")
    
    async with DatabaseSession() as db:
        saved_count = 0
        skipped_count = 0
        
        for post_data in posts:
            try:
                # Check if post already exists
                existing = await db.scalar(
                    select(Post).where(Post.post_id == post_data["post_id"])
                )
                
                if existing:
                    skipped_count += 1
                    continue
                
                # Create new post
                post = Post(
                    post_id=post_data["post_id"],
                    source=post_data["source"],
                    subreddit=post_data.get("subreddit"),
                    author=post_data["author"],
                    title=post_data["title"],
                    content=post_data.get("content", ""),
                    url=post_data.get("url"),
                    score=post_data["score"],
                    num_comments=post_data["num_comments"],
                    created_at=post_data["created_at"],
                    is_processed=False,
                    is_trending=False
                )
                
                db.add(post)
                saved_count += 1
            
            except Exception as e:
                print(f"❌ Error saving post {post_data['post_id']}: {e}")
        
        await db.commit()
    
    print(f"✅ Saved: {saved_count} | Skipped (duplicates): {skipped_count}")


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Main scraping workflow"""
    print("🤖 Reddit Scraper Starting...\n")
    
    # Initialize database
    await init_db()
    
    # Get Reddit client
    reddit = get_reddit_client()
    
    # Get configuration from .env
    subreddits_str = os.getenv("SUBREDDITS", "news,worldnews,technology")
    subreddits = [s.strip() for s in subreddits_str.split(",")]
    
    limit_per_sub = int(os.getenv("SCRAPE_LIMIT", "100"))
    
    # Option 1: Scrape from multiple subreddits
    print("\n📋 Scraping Configuration:")
    print(f"   Subreddits: {subreddits}")
    print(f"   Limit per subreddit: {limit_per_sub}")
    
    posts = scrape_multiple_subreddits(
        reddit=reddit,
        subreddits=subreddits,
        limit_per_sub=limit_per_sub,
        sort_by="hot"
    )
    
    # Option 2: Keyword-based scraping (optional)
    # keywords = os.getenv("KEYWORDS", "").split(",")
    # if keywords and keywords[0]:
    #     keyword_posts = scrape_with_keywords(
    #         reddit=reddit,
    #         subreddit_name="all",
    #         keywords=keywords,
    #         limit=50
    #     )
    #     posts.extend(keyword_posts)
    
    if posts:
        # Save to JSON (backup)
        save_to_json(posts)
        
        # Save to database
        await save_to_database(posts)
        
        print(f"\n🎉 Scraping complete! Total posts: {len(posts)}")
    else:
        print("\n⚠️  No posts scraped. Check your configuration.")


# ============================================
# CLI ENTRY POINT
# ============================================

if __name__ == "__main__":
    """
    Run directly: python scraper/reddit_scraper.py
    """
    asyncio.run(main())
