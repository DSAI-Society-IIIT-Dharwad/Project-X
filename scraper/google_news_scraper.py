"""
Google News Scraper
Scrapes news from Google News for comparison with Reddit
"""

from gnews import GNews
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from sqlalchemy import select

from backend.db import DatabaseSession, init_db
from backend.models import Post

# ============================================
# GOOGLE NEWS CLIENT
# ============================================

def get_google_news_client(period='7d', max_results=100):
    """
    Initialize Google News client
    
    Args:
        period: Time period ('1h', '1d', '7d', '1m', '1y')
        max_results: Max results per query
    """
    google_news = GNews(
        language='en',
        country='US',
        period=period,
        max_results=max_results
    )
    return google_news

# ============================================
# SCRAPING FUNCTIONS
# ============================================

def scrape_google_news_by_topic(
    topic: str,
    max_results: int = 50,
    period: str = '7d'
) -> List[Dict]:
    """
    Scrape Google News for a specific topic
    
    Args:
        topic: Search topic/keyword
        max_results: Maximum articles to fetch
        period: Time period
    
    Returns:
        List of article dictionaries
    """
    print(f"\n📰 Scraping Google News for: '{topic}'")
    print(f"   Period: {period}")
    print(f"   Max results: {max_results}\n")
    
    google_news = get_google_news_client(period=period, max_results=max_results)
    
    try:
        # Search for topic
        articles = google_news.get_news(topic)
        
        processed_articles = []
        
        for article in articles:
            # Extract article data
            article_data = {
                'post_id': f"gnews_{hash(article['url'])}",  # Unique ID
                'source': 'google_news',
                'subreddit': article.get('publisher', {}).get('title', 'Unknown'),
                'author': article.get('publisher', {}).get('title', 'Unknown'),
                'title': article.get('title', ''),
                'content': article.get('description', ''),
                'url': article.get('url', ''),
                'score': 0,  # Google News doesn't have upvotes
                'num_comments': 0,  # No comments
                'created_at': parse_google_date(article.get('published date', '')),
                'workflow': 'google_news',
                'search_query': topic
            }
            
            processed_articles.append(article_data)
        
        print(f"✅ Found {len(processed_articles)} articles from Google News")
        return processed_articles
    
    except Exception as e:
        print(f"❌ Error scraping Google News: {e}")
        return []


def parse_google_date(date_str: str) -> datetime:
    """Parse Google News date format"""
    try:
        # Google News dates are like: "Wed, 29 Oct 2025 12:00:00 GMT"
        from dateutil import parser
        return parser.parse(date_str)
    except:
        # Fallback to current time
        return datetime.now(timezone.utc)


def scrape_monitored_topics_google(
    topics: List[str],
    max_per_topic: int = 30
) -> List[Dict]:
    """
    Scrape multiple topics from Google News
    
    Args:
        topics: List of topics to search
        max_per_topic: Max articles per topic
    
    Returns:
        Combined list of all articles
    """
    print(f"\n📰 GOOGLE NEWS: Scraping {len(topics)} topics")
    
    all_articles = []
    
    for topic in topics:
        print(f"\n📥 Searching: {topic}")
        articles = scrape_google_news_by_topic(topic, max_results=max_per_topic)
        all_articles.extend(articles)
    
    print(f"\n✅ Total: {len(all_articles)} articles from Google News")
    return all_articles

# ============================================
# DATABASE STORAGE
# ============================================

async def save_google_news_to_db(articles: List[Dict]) -> Dict:
    """Save Google News articles to database"""
    print(f"\n💾 Saving {len(articles)} articles to database...")
    
    saved = 0
    skipped = 0
    
    async with DatabaseSession() as db:
        for article_data in articles:
            try:
                # Check if exists
                existing = await db.scalar(
                    select(Post).where(Post.post_id == article_data['post_id'])
                )
                
                if existing:
                    skipped += 1
                    continue
                
                # Create post
                post = Post(
                    post_id=article_data['post_id'],
                    source=article_data['source'],
                    subreddit=article_data['subreddit'],
                    author=article_data['author'],
                    title=article_data['title'],
                    content=article_data['content'],
                    url=article_data['url'],
                    score=article_data['score'],
                    num_comments=article_data['num_comments'],
                    created_at=article_data['created_at'],
                    is_processed=False,
                    is_trending=False
                )
                
                db.add(post)
                saved += 1
            
            except Exception as e:
                print(f"❌ Error saving article: {e}")
        
        await db.commit()
    
    print(f"✅ Saved: {saved} | Skipped: {skipped}")
    return {'saved': saved, 'skipped': skipped}


def save_to_json(articles: List[Dict], filename: Optional[str] = None):
    """Save articles to JSON"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"google_news_{timestamp}.json"
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, default=str)
    
    print(f"💾 Saved to: {filepath}")

# ============================================
# COMPARISON FUNCTIONS
# ============================================

async def compare_sentiments(topic: str) -> Dict:
    """
    Compare sentiment between Reddit and Google News for a topic
    
    Args:
        topic: Topic to compare
    
    Returns:
        Dictionary with comparison data
    """
    print(f"\n🔍 Comparing sentiments for: {topic}")
    
    async with DatabaseSession() as db:
        # Get Reddit posts
        from sqlalchemy import or_, func
        
        reddit_posts = await db.execute(
            select(Post).where(
                Post.source == 'reddit',
                or_(
                    func.lower(Post.title).contains(topic.lower()),
                    func.lower(Post.content).contains(topic.lower())
                )
            )
        )
        reddit_posts = reddit_posts.scalars().all()
        
        # Get Google News posts
        gnews_posts = await db.execute(
            select(Post).where(
                Post.source == 'google_news',
                or_(
                    func.lower(Post.title).contains(topic.lower()),
                    func.lower(Post.content).contains(topic.lower())
                )
            )
        )
        gnews_posts = gnews_posts.scalars().all()
        
        # Calculate sentiment stats
        def calc_sentiment(posts):
            if not posts:
                return {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0.0}
            
            pos = sum(1 for p in posts if p.sentiment_label == 'positive')
            neg = sum(1 for p in posts if p.sentiment_label == 'negative')
            neu = sum(1 for p in posts if p.sentiment_label == 'neutral')
            
            sentiments = [p.sentiment_score for p in posts if p.sentiment_score is not None]
            avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            return {
                'positive': pos,
                'negative': neg,
                'neutral': neu,
                'avg': avg,
                'total': len(posts)
            }
        
        reddit_sentiment = calc_sentiment(reddit_posts)
        gnews_sentiment = calc_sentiment(gnews_posts)
        
        # Calculate difference
        sentiment_diff = reddit_sentiment['avg'] - gnews_sentiment['avg']
        
        return {
            'topic': topic,
            'reddit': reddit_sentiment,
            'google_news': gnews_sentiment,
            'difference': sentiment_diff,
            'agreement': abs(sentiment_diff) < 0.2  # Similar if within 0.2
        }

# ============================================
# MAIN WORKFLOW
# ============================================

async def workflow_google_news(topics: List[str]) -> Dict:
    """
    Complete Google News workflow
    
    Args:
        topics: List of topics to scrape
    
    Returns:
        Stats dictionary
    """
    print("="*60)
    print("📰 GOOGLE NEWS WORKFLOW")
    print("="*60)
    
    await init_db()
    
    # Scrape
    articles = scrape_monitored_topics_google(topics)
    
    if articles:
        # Save to JSON
        save_to_json(articles)
        
        # Save to database
        stats = await save_google_news_to_db(articles)
        
        return {
            'success': True,
            'articles_found': len(articles),
            'articles_saved': stats['saved'],
            'articles_skipped': stats['skipped']
        }
    
    return {'success': False, 'error': 'No articles found'}

# ============================================
# CLI
# ============================================

async def main():
    """Main entry point"""
    # Example: Scrape common topics
    topics = [
        "artificial intelligence",
        "climate change",
        "SpaceX",
        "cryptocurrency",
        "politics"
    ]
    
    result = await workflow_google_news(topics)
    print(f"\n✅ Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
