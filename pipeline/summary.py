"""
Text Summarization using Google Gemini
Generates summaries for topics, trending posts, and daily digests
"""

import google.generativeai as genai
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import select, desc
from dotenv import load_dotenv

from backend.db import DatabaseSession
from backend.models import Post, Topic, SummaryCache

# Load environment
load_dotenv()

# ============================================
# GEMINI API SETUP
# ============================================

class GeminiSummarizer:
    """Wrapper for Google Gemini API"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Initialize Gemini API"""
        if self._model is None:
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env file")
            
            genai.configure(api_key=api_key)
            
            # Use Gemini Pro model
            self._model = genai.GenerativeModel('gemini-2.0-flash')
            
            print("✅ Gemini API initialized")
    
    @property
    def model(self):
        if self._model is None:
            self.initialize()
        return self._model
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text using Gemini
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
        
        Returns:
            Generated text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                )
            )
            
            return response.text
        
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return ""


# Global summarizer instance
summarizer = GeminiSummarizer()


# ============================================
# PROMPT TEMPLATES
# ============================================

TOPIC_SUMMARY_PROMPT = """You are a news analyst. Summarize the following posts about a topic.

Topic: {topic_name}
Keywords: {keywords}

Posts:
{posts_text}

Provide a concise 3-4 sentence summary covering:
1. Main theme/event
2. Key developments
3. Overall sentiment

Summary:"""


TRENDING_SUMMARY_PROMPT = """You are a news analyst. Summarize today's trending topics.

Trending Topics:
{topics_text}

Provide a brief overview (5-6 sentences) highlighting:
1. Most significant trends
2. Common themes
3. Notable sentiment patterns

Summary:"""


DAILY_DIGEST_PROMPT = """You are a news analyst. Create a daily digest of news posts.

Date: {date}

Top Stories:
{posts_text}

Create a comprehensive daily digest (8-10 sentences) including:
1. Major headlines
2. Key developments by category
3. Sentiment overview
4. Notable trends

Digest:"""


POST_SUMMARY_PROMPT = """Summarize this news post in 2-3 sentences:

Title: {title}
Content: {content}

Summary:"""


# ============================================
# SUMMARIZATION FUNCTIONS
# ============================================

def summarize_topic(topic_name: str, keywords: List[str], posts: List[Post]) -> str:
    """
    Generate summary for a topic
    
    Args:
        topic_name: Name of the topic
        keywords: Topic keywords
        posts: List of posts in topic
    
    Returns:
        Summary text
    """
    # Prepare posts text
    posts_text = []
    for i, post in enumerate(posts[:10], 1):  # Limit to top 10
        posts_text.append(f"{i}. {post.title}")
        if post.content and len(post.content) > 0:
            posts_text.append(f"   {post.content[:200]}...")
    
    posts_str = "\n".join(posts_text)
    
    # Create prompt
    prompt = TOPIC_SUMMARY_PROMPT.format(
        topic_name=topic_name,
        keywords=", ".join(keywords[:5]),
        posts_text=posts_str
    )
    
    # Generate summary
    print(f"📝 Generating summary for topic: {topic_name}")
    summary = summarizer.generate(prompt, max_tokens=300)
    
    return summary


def summarize_trending(topics_data: List[Dict]) -> str:
    """
    Generate summary of trending topics
    
    Args:
        topics_data: List of topic dictionaries with name, keywords, posts
    
    Returns:
        Summary text
    """
    # Prepare topics text
    topics_text = []
    for i, topic in enumerate(topics_data[:10], 1):
        topics_text.append(
            f"{i}. {topic['name']} ({topic['post_count']} posts)\n"
            f"   Keywords: {', '.join(topic['keywords'][:5])}\n"
            f"   Sentiment: {topic.get('avg_sentiment', 0):.2f}"
        )
    
    topics_str = "\n\n".join(topics_text)
    
    # Create prompt
    prompt = TRENDING_SUMMARY_PROMPT.format(topics_text=topics_str)
    
    # Generate summary
    print("📝 Generating trending topics summary")
    summary = summarizer.generate(prompt, max_tokens=400)
    
    return summary


def summarize_daily_digest(date: datetime, posts: List[Post]) -> str:
    """
    Generate daily news digest
    
    Args:
        date: Date for digest
        posts: Top posts of the day
    
    Returns:
        Digest text
    """
    # Prepare posts text
    posts_text = []
    for i, post in enumerate(posts[:20], 1):  # Top 20 posts
        sentiment = "😊" if post.sentiment_score and post.sentiment_score > 0.2 else "😐" if post.sentiment_score and post.sentiment_score > -0.2 else "😟"
        posts_text.append(
            f"{i}. [{sentiment}] {post.title}\n"
            f"   Score: {post.score} | Comments: {post.num_comments}"
        )
    
    posts_str = "\n\n".join(posts_text)
    
    # Create prompt
    prompt = DAILY_DIGEST_PROMPT.format(
        date=date.strftime("%Y-%m-%d"),
        posts_text=posts_str
    )
    
    # Generate digest
    print(f"📝 Generating daily digest for {date.date()}")
    digest = summarizer.generate(prompt, max_tokens=600)
    
    return digest


def summarize_single_post(post: Post) -> str:
    """
    Generate summary for a single post
    
    Args:
        post: Post to summarize
    
    Returns:
        Summary text
    """
    prompt = POST_SUMMARY_PROMPT.format(
        title=post.title,
        content=post.content[:500] if post.content else "No content"
    )
    
    summary = summarizer.generate(prompt, max_tokens=150)
    return summary


# ============================================
# CACHING SYSTEM
# ============================================

async def get_cached_summary(cache_key: str) -> Optional[str]:
    """
    Get summary from cache if exists and not expired
    
    Args:
        cache_key: Unique cache identifier
    
    Returns:
        Cached summary or None
    """
    async with DatabaseSession() as db:
        result = await db.execute(
            select(SummaryCache).where(SummaryCache.cache_key == cache_key)
        )
        cached = result.scalar_one_or_none()
        
        if cached:
            # Check expiration
            if cached.expires_at and cached.expires_at < datetime.utcnow():
                # Expired
                await db.delete(cached)
                await db.commit()
                return None
            
            print(f"✅ Using cached summary: {cache_key}")
            return cached.summary_text
        
        return None


async def save_summary_to_cache(
    cache_key: str,
    summary_type: str,
    summary_text: str,
    expires_hours: int = 24
):
    """
    Save summary to cache
    
    Args:
        cache_key: Unique identifier
        summary_type: Type of summary
        summary_text: The summary
        expires_hours: Hours until expiration
    """
    async with DatabaseSession() as db:
        # Delete existing if any
        existing = await db.scalar(
            select(SummaryCache).where(SummaryCache.cache_key == cache_key)
        )
        if existing:
            await db.delete(existing)
        
        # Create new cache entry
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        cache = SummaryCache(
            cache_key=cache_key,
            summary_type=summary_type,
            summary_text=summary_text,
            expires_at=expires_at,
            token_count=len(summary_text.split())
        )
        
        db.add(cache)
        await db.commit()
        
        print(f"💾 Cached summary: {cache_key}")


# ============================================
# HIGH-LEVEL SUMMARY GENERATION
# ============================================

async def generate_topic_summary(topic_id: int, use_cache: bool = True) -> str:
    """
    Generate or retrieve cached topic summary
    
    Args:
        topic_id: Topic database ID
        use_cache: Whether to use cached version
    
    Returns:
        Summary text
    """
    cache_key = f"topic_{topic_id}"
    
    # Check cache
    if use_cache:
        cached = await get_cached_summary(cache_key)
        if cached:
            return cached
    
    # Generate new summary
    async with DatabaseSession() as db:
        # Get topic
        topic = await db.scalar(select(Topic).where(Topic.id == topic_id))
        
        if not topic:
            return "Topic not found"
        
        # Get posts
        posts_result = await db.execute(
            select(Post).where(Post.topic_id == topic_id)
            .order_by(desc(Post.score))
            .limit(10)
        )
        posts = posts_result.scalars().all()
        
        if not posts:
            return "No posts found for this topic"
        
        # Generate summary
        summary = summarize_topic(topic.name, topic.keywords, posts)
        
        # Cache it
        await save_summary_to_cache(cache_key, "topic", summary, expires_hours=12)
        
        return summary


async def generate_trending_summary(use_cache: bool = True) -> str:
    """
    Generate summary of current trending topics
    
    Args:
        use_cache: Whether to use cached version
    
    Returns:
        Summary text
    """
    today = datetime.utcnow().date()
    cache_key = f"trending_{today}"
    
    # Check cache
    if use_cache:
        cached = await get_cached_summary(cache_key)
        if cached:
            return cached
    
    # Generate new summary
    async with DatabaseSession() as db:
        # Get trending topics
        topics_result = await db.execute(
            select(Topic).order_by(desc(Topic.num_posts)).limit(10)
        )
        topics = topics_result.scalars().all()
        
        topics_data = [
            {
                'name': t.name,
                'keywords': t.keywords,
                'post_count': t.num_posts,
                'avg_sentiment': t.avg_sentiment or 0.0
            }
            for t in topics
        ]
        
        # Generate summary
        summary = summarize_trending(topics_data)
        
        # Cache it
        await save_summary_to_cache(cache_key, "trending", summary, expires_hours=6)
        
        return summary


async def generate_daily_digest(date: Optional[datetime] = None, use_cache: bool = True) -> str:
    """
    Generate daily news digest
    
    Args:
        date: Date for digest (default: today)
        use_cache: Whether to use cached version
    
    Returns:
        Digest text
    """
    if date is None:
        date = datetime.utcnow()
    
    cache_key = f"daily_{date.date()}"
    
    # Check cache
    if use_cache:
        cached = await get_cached_summary(cache_key)
        if cached:
            return cached
    
    # Generate new digest
    async with DatabaseSession() as db:
        # Get posts from the day
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        posts_result = await db.execute(
            select(Post).where(
                Post.created_at >= start_time,
                Post.created_at < end_time
            ).order_by(desc(Post.score)).limit(20)
        )
        posts = posts_result.scalars().all()
        
        if not posts:
            return "No posts found for this date"
        
        # Generate digest
        digest = summarize_daily_digest(date, posts)
        
        # Cache it
        await save_summary_to_cache(cache_key, "daily", digest, expires_hours=24)
        
        return digest


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Generate summaries for topics and trending"""
    print("🚀 Starting summarization pipeline...\n")
    
    # Generate trending summary
    print("1️⃣ Generating trending topics summary...")
    trending_summary = await generate_trending_summary(use_cache=False)
    print(f"\n📰 Trending Summary:\n{trending_summary}\n")
    
    # Generate daily digest
    print("\n2️⃣ Generating daily digest...")
    daily_digest = await generate_daily_digest(use_cache=False)
    print(f"\n📰 Daily Digest:\n{daily_digest}\n")
    
    # Generate summaries for top 3 topics
    print("\n3️⃣ Generating topic summaries...")
    async with DatabaseSession() as db:
        topics_result = await db.execute(
            select(Topic).order_by(desc(Topic.num_posts)).limit(3)
        )
        topics = topics_result.scalars().all()
        
        for topic in topics:
            print(f"\n📌 Topic: {topic.name}")
            summary = await generate_topic_summary(topic.id, use_cache=False)
            print(f"   {summary}\n")
    
    print("✅ Summarization pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
