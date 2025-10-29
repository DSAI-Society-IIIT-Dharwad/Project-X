"""
FastAPI Backend for AI News Assistant
Combines Reddit scraping, sentiment analysis, and chat functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc, or_
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.db import DatabaseSession
from backend.models import Post, Topic
from config import MONITORED_SUBREDDITS, GOOGLE_NEWS_TOPICS
from scraper.reddit_scraper import workflow_1_refresh, workflow_2_query
from scraper.google_news_scraper import workflow_google_news, compare_sentiments
from backend.ingest import process_unprocessed_posts
from pipeline.embeddings import build_faiss_index
from pipeline.topic_model import train_topic_model
from pipeline.quick_pipeline import process_query_results
from backend.chat_storage import save_chat_interaction, get_chat_history
from pipeline.summary import GeminiSummarizer
import uuid
import time

app = FastAPI(title="AI News Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
gemini_summarizer = GeminiSummarizer()
gemini_summarizer.initialize()

# ============================================
# MODELS
# ============================================

class ChatRequest(BaseModel):
    query: str
    session_id: str

class SentimentRequest(BaseModel):
    topic: str

class CompareRequest(BaseModel):
    topic: str

# ============================================
# DASHBOARD ENDPOINTS
# ============================================

@app.get("/api/stats")
async def get_stats():
    """Get overall statistics for both sources"""
    async with DatabaseSession() as db:
        # Reddit stats
        reddit_total = await db.scalar(
            select(func.count(Post.id)).where(Post.source == 'reddit')
        ) or 0
        
        reddit_24h = await db.scalar(
            select(func.count(Post.id)).where(
                Post.source == 'reddit',
                Post.created_at >= datetime.utcnow() - timedelta(hours=24)
            )
        ) or 0
        
        reddit_sentiment = await db.scalar(
            select(func.avg(Post.sentiment_score)).where(Post.source == 'reddit')
        ) or 0.0
        
        # Google News stats
        gnews_total = await db.scalar(
            select(func.count(Post.id)).where(Post.source == 'google_news')
        ) or 0
        
        gnews_24h = await db.scalar(
            select(func.count(Post.id)).where(
                Post.source == 'google_news',
                Post.created_at >= datetime.utcnow() - timedelta(hours=24)
            )
        ) or 0
        
        gnews_sentiment = await db.scalar(
            select(func.avg(Post.sentiment_score)).where(Post.source == 'google_news')
        ) or 0.0
        
        return {
            'reddit': {
                'total': reddit_total,
                'recent_24h': reddit_24h,
                'avg_sentiment': float(reddit_sentiment)
            },
            'google_news': {
                'total': gnews_total,
                'recent_24h': gnews_24h,
                'avg_sentiment': float(gnews_sentiment)
            }
        }

@app.get("/api/sentiment-distribution/{source}")
async def get_sentiment_distribution(source: str):
    """Get sentiment distribution for a source"""
    async with DatabaseSession() as db:
        positive = await db.scalar(
            select(func.count(Post.id)).where(
                Post.source == source,
                Post.sentiment_label == 'positive'
            )
        ) or 0
        
        negative = await db.scalar(
            select(func.count(Post.id)).where(
                Post.source == source,
                Post.sentiment_label == 'negative'
            )
        ) or 0
        
        neutral = await db.scalar(
            select(func.count(Post.id)).where(
                Post.source == source,
                Post.sentiment_label == 'neutral'
            )
        ) or 0
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }

@app.get("/api/recent-posts/{source}")
async def get_recent_posts(source: str, limit: int = 10):
    """Get recent posts from source"""
    async with DatabaseSession() as db:
        result = await db.execute(
            select(Post).where(Post.source == source)
            .order_by(desc(Post.created_at))
            .limit(limit)
        )
        posts = result.scalars().all()
        
        return [
            {
                'id': p.id,
                'title': p.title,
                'subreddit': p.subreddit or 'N/A',
                'sentiment': p.sentiment_label or 'neutral',
                'sentiment_score': p.sentiment_score or 0.0,
                'score': p.score,
                'url': p.url,
                'created_at': p.created_at.isoformat()
            }
            for p in posts
        ]

@app.post("/api/refresh")
async def refresh_data():
    """Refresh both Reddit and Google News data"""
    try:
        # Reddit
        reddit_result = await workflow_1_refresh()
        
        # Google News
        gnews_result = await workflow_google_news(GOOGLE_NEWS_TOPICS)
        
        # Process
        await process_unprocessed_posts(limit=None)
        
        # Build index
        await build_faiss_index()
        
        # Topics
        async with DatabaseSession() as db:
            total = await db.scalar(select(func.count(Post.id)))
            if total > 100:
                await train_topic_model(min_posts=50)
        
        return {
            'success': True,
            'reddit': {
                'posts_saved': reddit_result['posts_saved']
            },
            'google_news': {
                'articles_saved': gnews_result['articles_saved']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-sentiment")
async def compare_topic_sentiment(request: CompareRequest):
    """Compare sentiment between Reddit and Google News for a topic"""
    try:
        comparison = await compare_sentiments(request.topic)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Intelligent chat that searches Reddit and answers questions"""
    try:
        start_time = time.time()
        
        # Step 1: Search Reddit
        scrape_result = await workflow_2_query(request.query, search_all=True)
        
        if not scrape_result['success'] or scrape_result['posts_found'] == 0:
            return {
                'answer': f"I searched Reddit but couldn't find recent posts about '{request.query}'. This topic might not be actively discussed right now, or try rephrasing your question.",
                'posts_found': 0,
                'analysis': None,
                'confidence': 0.0,
                'posts': []
            }
        
        # Step 2: Get posts
        async with DatabaseSession() as db:
            scraped_post_ids = [p['post_id'] for p in scrape_result['posts'][:50]]
            
            result = await db.execute(
                select(Post).where(Post.post_id.in_(scraped_post_ids))
            )
            posts = result.scalars().all()
            post_ids = [p.id for p in posts]
        
        if not posts:
            await asyncio.sleep(2)
            async with DatabaseSession() as db:
                result = await db.execute(
                    select(Post).where(Post.post_id.in_(scraped_post_ids))
                )
                posts = result.scalars().all()
                post_ids = [p.id for p in posts]
        
        if not posts:
            return {
                'answer': f"Found {scrape_result['posts_found']} posts but couldn't process them.",
                'posts_found': scrape_result['posts_found'],
                'analysis': None,
                'confidence': 0.0,
                'posts': []
            }
        
        # Step 3: Analyze
        analysis = await process_query_results(post_ids)
        
        # Step 4: Generate answer
        context = _build_context(posts, analysis)
        
        prompt = f"""You are a helpful news assistant that answers questions based on recent Reddit discussions.

User Question: {request.query}

Recent Reddit Posts Context:
{context}

Analysis Summary:
- Total posts found: {analysis['total_posts']}
- Average sentiment: {analysis['sentiment']['average_score']:.2f}
- Top keywords: {', '.join(analysis['keywords'][:8])}
- Main subreddits: {', '.join(list(analysis['subreddits'].keys())[:5])}

Instructions:
1. Answer the user's question directly based on the Reddit posts
2. Be conversational and friendly (3-5 sentences)
3. Mention specific trends or sentiments if relevant
4. If sentiment is notable, mention it
5. Keep it concise but informative

Answer:"""
        
        answer = gemini_summarizer.generate(prompt, max_tokens=400)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        confidence = min(analysis['total_posts'] / 20.0, 1.0)
        
        # Save interaction
        await save_chat_interaction(
            user_query=request.query,
            bot_response=answer,
            relevant_post_ids=post_ids[:10],
            confidence_score=confidence,
            session_id=request.session_id,
            response_time_ms=response_time_ms,
            query_type="intelligent_search"
        )
        
        return {
            'answer': answer,
            'posts_found': analysis['total_posts'],
            'analysis': analysis,
            'confidence': confidence,
            'posts': [
                {
                    'title': p.title,
                    'subreddit': p.subreddit,
                    'sentiment': p.sentiment_label,
                    'score': p.score,
                    'url': p.url
                }
                for p in posts[:5]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _build_context(posts, analysis):
    """Build context from posts"""
    context_parts = []
    for i, post in enumerate(posts[:8], 1):
        sentiment = f"[{post.sentiment_label}]" if post.sentiment_label else ""
        context_parts.append(f"{i}. {sentiment} r/{post.subreddit}: {post.title}")
    return "\n".join(context_parts)

@app.get("/api/chat-history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """Get chat history for session"""
    history = await get_chat_history(session_id, limit=limit)
    return [
        {
            'user_query': h['user_query'],
            'bot_response': h['bot_response'],
            'confidence': h['confidence'],
            'created_at': h['created_at'].isoformat()
        }
        for h in history
    ]

# ============================================
# SENTIMENT ANALYSIS ENDPOINTS
# ============================================

@app.post("/api/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Full sentiment analysis for a topic"""
    try:
        # Search Reddit
        scrape_result = await workflow_2_query(request.topic, search_all=True)
        
        if not scrape_result['success'] or scrape_result['posts_found'] == 0:
            return {'success': False, 'message': 'No posts found'}
        
        # Get posts
        async with DatabaseSession() as db:
            search_terms = request.topic.lower().split()
            conditions = []
            for term in search_terms:
                conditions.append(or_(
                    func.lower(Post.title).contains(term),
                    func.lower(Post.content).contains(term)
                ))
            
            result = await db.execute(
                select(Post).where(or_(*conditions))
                .order_by(Post.created_at.desc())
                .limit(100)
            )
            posts = result.scalars().all()
            post_ids = [p.id for p in posts]
        
        # Analyze
        analysis = await process_query_results(post_ids)
        
        # Additional metrics
        sentiment_by_subreddit = {}
        sentiment_over_time = {}
        
        for post in posts:
            # By subreddit
            if post.subreddit not in sentiment_by_subreddit:
                sentiment_by_subreddit[post.subreddit] = {
                    'pos': 0, 'neg': 0, 'neu': 0, 'total': 0
                }
            
            sentiment_by_subreddit[post.subreddit]['total'] += 1
            if post.sentiment_label == 'positive':
                sentiment_by_subreddit[post.subreddit]['pos'] += 1
            elif post.sentiment_label == 'negative':
                sentiment_by_subreddit[post.subreddit]['neg'] += 1
            else:
                sentiment_by_subreddit[post.subreddit]['neu'] += 1
            
            # Over time
            date = post.created_at.date().isoformat()
            if date not in sentiment_over_time:
                sentiment_over_time[date] = {'pos': 0, 'neg': 0, 'neu': 0}
            
            if post.sentiment_label == 'positive':
                sentiment_over_time[date]['pos'] += 1
            elif post.sentiment_label == 'negative':
                sentiment_over_time[date]['neg'] += 1
            else:
                sentiment_over_time[date]['neu'] += 1
        
        return {
            'success': True,
            'analysis': analysis,
            'sentiment_by_subreddit': sentiment_by_subreddit,
            'sentiment_over_time': sentiment_over_time,
            'sample_posts': [
                {
                    'title': p.title,
                    'subreddit': p.subreddit,
                    'sentiment': p.sentiment_label,
                    'sentiment_score': p.sentiment_score,
                    'score': p.score,
                    'num_comments': p.num_comments,
                    'url': p.url
                }
                for p in posts[:20]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "AI News Assistant API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)