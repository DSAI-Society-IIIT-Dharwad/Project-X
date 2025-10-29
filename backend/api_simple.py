"""
Simplified FastAPI Backend for AI News Assistant
Basic version without complex ML dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.db import DatabaseSession
from backend.models import Post, Topic
from sqlalchemy import select, func, desc

app = FastAPI(title="AI News Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # For now, return mock data
        return {
            'success': True,
            'reddit': {
                'posts_saved': 0
            },
            'google_news': {
                'articles_saved': 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-sentiment")
async def compare_topic_sentiment(request: CompareRequest):
    """Compare sentiment between Reddit and Google News for a topic"""
    try:
        # Mock comparison for now
        return {
            'topic': request.topic,
            'reddit_sentiment': 0.2,
            'news_sentiment': 0.1,
            'difference': 0.1,
            'analysis': 'Mock analysis - sentiments are similar'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Intelligent chat that searches Reddit and answers questions"""
    try:
        # Mock response for now
        return {
            'answer': f"I searched for '{request.query}' but this is a mock response. The full system will be implemented with real Reddit search and AI analysis.",
            'posts_found': 0,
            'analysis': {
                'total_posts': 0,
                'sentiment': {'average_score': 0.0},
                'keywords': [],
                'subreddits': {}
            },
            'confidence': 0.0,
            'posts': []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat-history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """Get chat history for session"""
    # Mock history for now
    return []

# ============================================
# SENTIMENT ANALYSIS ENDPOINTS
# ============================================

@app.post("/api/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Full sentiment analysis for a topic"""
    try:
        # Mock analysis for now
        return {
            'success': True,
            'analysis': {
                'total_posts': 0,
                'sentiment': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'average_score': 0.0
                },
                'keywords': [],
                'subreddits': {}
            },
            'sentiment_by_subreddit': {},
            'sentiment_over_time': {},
            'sample_posts': []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "AI News Assistant API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
