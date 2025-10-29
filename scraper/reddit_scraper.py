"""
Enhanced Reddit Scraper with Query Support
Supports both monitored subreddits and dynamic query-based search
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
from config import MONITORED_SUBREDDITS, POSTS_PER_SUBREDDIT, QUERY_SEARCH_LIMIT, QUERY_TIME_FILTER

load_dotenv()

# ============================================
# REDDIT CLIENT
# ============================================

def get_reddit_client():
    """Initialize and return Reddit client"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD")
        )
        print(f"✅ Authenticated as: {reddit.user.me()}")
        return reddit
    except Exception as e:
        print(f"❌ Reddit authentication failed: {e}")
        raise

# ============================================
# WORKFLOW 1: MONITORED SUBREDDITS
# ============================================

def scrape_monitored_subreddits(
    reddit: praw.Reddit,
    subreddits: Optional[List[str]] = None,
    limit_per_sub: int = None
) -> List[Dict]:
    """
    Scrape from monitored subreddits (Workflow 1)
    
    Args:
        reddit: PRAW Reddit instance
        subreddits: List of subreddits (defaults to config)
        limit_per_sub: Posts per subreddit (defaults to config)
    
    Returns:
        List of post dictionaries
    """
    if subreddits is None:
        subreddits = MONITORED_SUBREDDITS
    
    if limit_per_sub is None:
        limit_per_sub = POSTS_PER_SUBREDDIT
    
    print(f"\n📡 WORKFLOW 1: Scraping monitored subreddits")
    print(f"   Subreddits: {subreddits}")
    print(f"   Posts per subreddit: {limit_per_sub}\n")
    
    all_posts = []
    
    for subreddit_name in subreddits:
        try:
            print(f"📥 r/{subreddit_name}...", end=" ")
            subreddit = reddit.subreddit(subreddit_name)
            posts = subreddit.hot(limit=limit_per_sub)
            
            count = 0
            for submission in posts:
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
                    "workflow": "monitored"  # Mark source
                }
                
                all_posts.append(post_data)
                count += 1
            
            print(f"✅ {count} posts")
            time.sleep(2)  # Rate limiting
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Total: {len(all_posts)} posts from monitored subreddits")
    return all_posts

# ============================================
# WORKFLOW 2: QUERY-BASED SEARCH
# ============================================

def search_reddit_by_query(
    reddit: praw.Reddit,
    query: str,
    limit: int = None,
    time_filter: str = None,
    search_all_reddit: bool = True  # NEW PARAMETER
) -> List[Dict]:
    """
    Search Reddit for specific query (Workflow 2)
    
    Args:
        reddit: PRAW Reddit instance
        query: Search query string
        limit: Max results
        time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
        search_all_reddit: If True, search ALL of Reddit (not just monitored subs)
    
    Returns:
        List of matching posts
    """
    if limit is None:
        limit = QUERY_SEARCH_LIMIT
    
    if time_filter is None:
        time_filter = QUERY_TIME_FILTER
    
    print(f"\n🔍 WORKFLOW 2: Query-based search")
    print(f"   Query: '{query}'")
    print(f"   Scope: {'ALL REDDIT' if search_all_reddit else 'Monitored subreddits'}")
    print(f"   Time filter: {time_filter}")
    print(f"   Limit: {limit}\n")
    
    all_posts = []
    
    if search_all_reddit:
        # SEARCH ALL OF REDDIT (like reddit.com/search)
        try:
            print(f"🌍 Searching ALL of Reddit...", end=" ")
            
            # Search across all subreddits
            search_results = reddit.subreddit("all").search(
                query=query,
                limit=limit,
                time_filter=time_filter,
                sort='relevance'
            )
            
            count = 0
            for submission in search_results:
                # Skip NSFW and certain subreddits if needed
                if submission.over_18:
                    continue
                
                post_data = {
                    "post_id": submission.id,
                    "source": "reddit",
                    "subreddit": submission.subreddit.display_name,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "title": submission.title,
                    "content": submission.selftext if submission.selftext else "",
                    "url": submission.url,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_at": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "workflow": "query",
                    "search_query": query
                }
                
                all_posts.append(post_data)
                count += 1
            
            print(f"✅ {count} results")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        # SEARCH ONLY MONITORED SUBREDDITS (old behavior)
        subreddits = MONITORED_SUBREDDITS
        
        for subreddit_name in subreddits:
            try:
                print(f"🔎 Searching r/{subreddit_name}...", end=" ")
                subreddit = reddit.subreddit(subreddit_name)
                
                search_results = subreddit.search(
                    query=query,
                    limit=limit // len(subreddits),  # Distribute limit
                    time_filter=time_filter,
                    sort='relevance'
                )
                
                count = 0
                for submission in search_results:
                    post_data = {
                        "post_id": submission.id,
                        "source": "reddit",
                        "subreddit": submission.subreddit.display_name,
                        "author": str(submission.author) if submission.author else "[deleted]",
                        "title": submission.title,
                        "content": submission.selftext if submission.selftext else "",
                        "url": submission.url,
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "created_at": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        "permalink": f"https://reddit.com{submission.permalink}",
                        "workflow": "query",
                        "search_query": query
                    }
                    
                    all_posts.append(post_data)
                    count += 1
                
                print(f"✅ {count} results")
                time.sleep(2)  # Rate limiting
            
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Remove duplicates
    seen_ids = set()
    unique_posts = []
    for post in all_posts:
        if post['post_id'] not in seen_ids:
            seen_ids.add(post['post_id'])
            unique_posts.append(post)
    
    print(f"\n✅ Total: {len(unique_posts)} unique posts for query '{query}'")
    return unique_posts

# ============================================
# DATA STORAGE
# ============================================

def save_to_json(posts: List[Dict], filename: Optional[str] = None, workflow: str = "monitored"):
    """Save scraped posts to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_{workflow}_{timestamp}.json"
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, default=str)
    
    print(f"💾 Saved to: {filepath}")
    return filepath

async def save_to_database(posts: List[Dict], workflow: str = "monitored"):
    """Save scraped posts to database"""
    print(f"\n💾 Saving {len(posts)} posts to database ({workflow})...")
    
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
    return {'saved': saved_count, 'skipped': skipped_count}

# ============================================
# MAIN FUNCTIONS
# ============================================

async def workflow_1_refresh():
    """
    Workflow 1: Refresh monitored subreddits
    Called when user clicks "Refresh" on dashboard
    """
    print("="*60)
    print("🔄 WORKFLOW 1: Refreshing News Dashboard")
    print("="*60)
    
    await init_db()
    reddit = get_reddit_client()
    
    # Scrape monitored subreddits
    posts = scrape_monitored_subreddits(reddit)
    
    if posts:
        # Save to JSON backup
        save_to_json(posts, workflow="monitored")
        
        # Save to database
        stats = await save_to_database(posts, workflow="monitored")
        
        return {
            'success': True,
            'posts_scraped': len(posts),
            'posts_saved': stats['saved'],
            'posts_skipped': stats['skipped']
        }
    
    return {'success': False, 'error': 'No posts scraped'}

async def workflow_2_query(query: str, search_all: bool = True):
    """
    Workflow 2: Search Reddit for query
    Called when user submits a query
    
    Args:
        query: User's search query
        search_all: If True, search ALL of Reddit (default)
    
    Returns:
        Dictionary with scraped posts and stats
    """
    print("="*60)
    print(f"🔍 WORKFLOW 2: Query Analysis")
    print("="*60)
    
    await init_db()
    reddit = get_reddit_client()
    
    # Search Reddit (ALL or monitored)
    posts = search_reddit_by_query(reddit, query, search_all_reddit=search_all)
    
    if posts:
        # Save to JSON backup
        query_safe = query.replace(' ', '_')[:30]
        save_to_json(posts, filename=f"query_{query_safe}.json", workflow="query")
        
        # Save to database
        stats = await save_to_database(posts, workflow="query")
        
        return {
            'success': True,
            'query': query,
            'posts_found': len(posts),
            'posts_saved': stats['saved'],
            'posts_skipped': stats['skipped'],
            'posts': posts
        }
    
    return {'success': False, 'error': f'No posts found for query: {query}'}

# ============================================
# CLI ENTRY POINTS
# ============================================

async def main():
    """Default: Run Workflow 1"""
    result = await workflow_1_refresh()
    print("\n" + "="*60)
    print(f"✅ Workflow 1 Complete: {result}")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line query
        query = ' '.join(sys.argv[1:])
        result = asyncio.run(workflow_2_query(query))
    else:
        # Default: monitored subreddits
        asyncio.run(main())
