"""
Ultra-Simple FastAPI Backend for AI News Assistant
No database dependencies - just mock data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import random
from datetime import datetime

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
    return {
        'reddit': {
            'total': random.randint(1000, 5000),
            'recent_24h': random.randint(50, 200),
            'avg_sentiment': round(random.uniform(-0.5, 0.5), 3)
        },
        'google_news': {
            'total': random.randint(500, 2000),
            'recent_24h': random.randint(20, 100),
            'avg_sentiment': round(random.uniform(-0.3, 0.3), 3)
        }
    }

@app.get("/api/sentiment-distribution/{source}")
async def get_sentiment_distribution(source: str):
    """Get sentiment distribution for a source"""
    total = random.randint(100, 1000)
    positive = random.randint(20, total // 3)
    negative = random.randint(20, total // 3)
    neutral = total - positive - negative
    
    return {
        'positive': positive,
        'negative': negative,
        'neutral': neutral
    }

@app.get("/api/recent-posts/{source}")
async def get_recent_posts(source: str, limit: int = 10):
    """Get recent posts from source"""
    posts = []
    for i in range(min(limit, 10)):
        posts.append({
            'id': i + 1,
            'title': f'Sample {source} post {i + 1} - This is a mock post for testing',
            'subreddit': f'r/sample{i + 1}' if source == 'reddit' else 'N/A',
            'sentiment': random.choice(['positive', 'negative', 'neutral']),
            'sentiment_score': round(random.uniform(-1, 1), 2),
            'score': random.randint(10, 1000),
            'url': f'https://example.com/post/{i + 1}',
            'created_at': datetime.now().isoformat()
        })
    
    return posts

@app.post("/api/refresh")
async def refresh_data():
    """Refresh both Reddit and Google News data"""
    return {
        'success': True,
        'reddit': {
            'posts_saved': random.randint(10, 50)
        },
        'google_news': {
            'articles_saved': random.randint(5, 25)
        }
    }

@app.post("/api/compare-sentiment")
async def compare_topic_sentiment(request: CompareRequest):
    """Compare sentiment between Reddit and Google News for a topic"""
    reddit_sentiment = round(random.uniform(-0.5, 0.5), 2)
    news_sentiment = round(random.uniform(-0.3, 0.3), 2)
    difference = abs(reddit_sentiment - news_sentiment)
    
    return {
        'topic': request.topic,
        'reddit_sentiment': reddit_sentiment,
        'news_sentiment': news_sentiment,
        'difference': difference,
        'analysis': f'Mock analysis for {request.topic} - sentiments differ by {difference:.2f}'
    }

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Intelligent chat that searches Reddit and answers questions"""
    posts_found = random.randint(5, 50)
    confidence = min(posts_found / 20.0, 1.0)
    
    return {
        'answer': f"I searched for '{request.query}' and found {posts_found} posts. This is a mock response - the full system will include real Reddit search and AI analysis. Based on the posts, here's what I found: [Mock analysis would go here]",
        'posts_found': posts_found,
        'analysis': {
            'total_posts': posts_found,
            'sentiment': {'average_score': round(random.uniform(-0.5, 0.5), 2)},
            'keywords': ['mock', 'keyword', 'analysis', 'test'],
            'subreddits': {'r/sample1': 5, 'r/sample2': 3, 'r/sample3': 2}
        },
        'confidence': confidence,
        'posts': [
            {
                'title': f'Sample post about {request.query}',
                'subreddit': 'r/sample',
                'sentiment': random.choice(['positive', 'negative', 'neutral']),
                'score': random.randint(10, 500),
                'url': 'https://example.com/post/1'
            }
            for _ in range(min(5, posts_found))
        ]
    }

@app.get("/api/chat-history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """Get chat history for session"""
    return [
        {
            'user_query': 'Sample question',
            'bot_response': 'Sample response',
            'confidence': 0.8,
            'created_at': datetime.now().isoformat()
        }
    ]

# ============================================
# SENTIMENT ANALYSIS ENDPOINTS
# ============================================

@app.post("/api/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Full sentiment analysis for a topic"""
    total_posts = random.randint(50, 500)
    positive = random.randint(10, total_posts // 3)
    negative = random.randint(10, total_posts // 3)
    neutral = total_posts - positive - negative
    
    return {
        'success': True,
        'analysis': {
            'total_posts': total_posts,
            'sentiment': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'average_score': round(random.uniform(-0.5, 0.5), 2)
            },
            'keywords': ['mock', 'keyword', 'analysis', 'test', 'sample'],
            'subreddits': {'r/sample1': 20, 'r/sample2': 15, 'r/sample3': 10}
        },
        'sentiment_by_subreddit': {
            'r/sample1': {'pos': 8, 'neg': 5, 'neu': 7, 'total': 20},
            'r/sample2': {'pos': 6, 'neg': 4, 'neu': 5, 'total': 15},
            'r/sample3': {'pos': 4, 'neg': 3, 'neu': 3, 'total': 10}
        },
        'sentiment_over_time': {
            '2024-01-01': {'pos': 5, 'neg': 3, 'neu': 4},
            '2024-01-02': {'pos': 7, 'neg': 2, 'neu': 6},
            '2024-01-03': {'pos': 4, 'neg': 5, 'neu': 3}
        },
        'sample_posts': [
            {
                'title': f'Sample post about {request.topic}',
                'subreddit': 'r/sample',
                'sentiment': random.choice(['positive', 'negative', 'neutral']),
                'sentiment_score': round(random.uniform(-1, 1), 2),
                'score': random.randint(10, 500),
                'num_comments': random.randint(5, 100),
                'url': f'https://example.com/post/{i}'
            }
            for i in range(1, 21)
        ]
    }

@app.get("/")
async def root():
    return {"message": "AI News Assistant API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
