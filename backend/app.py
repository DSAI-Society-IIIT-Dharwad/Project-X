"""
FastAPI Backend Application
Main API server for News Bot
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional
from datetime import datetime, timedelta

from backend.db import get_db, init_db, close_db, check_connection
from backend.models import Post, Topic, TrendScore, TopicDrift, SummaryCache
from backend.schemas import (
    PostResponse, PostCreate, PostUpdate, PostWithTopic,
    TopicResponse, TopicWithPosts,
    TrendingPost, TrendScoreResponse,
    DashboardStats, SentimentStats, TopicStats,
    ChatRequest, ChatResponse,
    ScraperConfig, ScraperStatus
)

# ============================================
# APP INITIALIZATION
# ============================================

app = FastAPI(
    title="NewsBot API",
    description="Real-time news trend analysis and sentiment chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# STARTUP & SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and check connections on startup"""
    print("🚀 Starting NewsBot API...")
    await init_db()
    await check_connection()
    print("✅ API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown"""
    print("👋 Shutting down NewsBot API...")
    await close_db()
    print("✅ Cleanup complete!")


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
async def root():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "service": "NewsBot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check with database status"""
    try:
        # Check database connection
        await db.execute(select(1))
        
        # Get counts
        post_count = await db.scalar(select(func.count(Post.id)))
        topic_count = await db.scalar(select(func.count(Topic.id)))
        
        return {
            "status": "healthy",
            "database": "connected",
            "posts": post_count,
            "topics": topic_count,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }


# ============================================
# POST ENDPOINTS
# ============================================

@app.get("/posts", response_model=List[PostResponse])
async def get_posts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    source: Optional[str] = Query(None, description="Filter by 'reddit' or 'twitter'"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment label"),
    is_trending: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get all posts with optional filters"""
    query = select(Post)
    
    # Apply filters
    if source:
        query = query.where(Post.source == source)
    if sentiment:
        query = query.where(Post.sentiment_label == sentiment)
    if is_trending is not None:
        query = query.where(Post.is_trending == is_trending)
    
    # Add pagination and ordering
    query = query.order_by(desc(Post.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    posts = result.scalars().all()
    
    return posts


@app.get("/posts/{post_id}", response_model=PostWithTopic)
async def get_post(post_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific post by ID with topic details"""
    query = select(Post).where(Post.id == post_id)
    result = await db.execute(query)
    post = result.scalar_one_or_none()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Get topic details if exists
    response = PostWithTopic.model_validate(post)
    if post.topic:
        response.topic_name = post.topic.name
        response.topic_keywords = post.topic.keywords
    
    return response


@app.post("/posts", response_model=PostResponse, status_code=201)
async def create_post(post: PostCreate, db: AsyncSession = Depends(get_db)):
    """Create a new post"""
    # Check if post already exists
    existing = await db.scalar(select(Post).where(Post.post_id == post.post_id))
    if existing:
        raise HTTPException(status_code=400, detail="Post already exists")
    
    # Create new post
    db_post = Post(**post.model_dump())
    db.add(db_post)
    await db.commit()
    await db.refresh(db_post)
    
    return db_post


@app.patch("/posts/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: int,
    post_update: PostUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update an existing post"""
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Update fields
    for field, value in post_update.model_dump(exclude_unset=True).items():
        setattr(post, field, value)
    
    await db.commit()
    await db.refresh(post)
    
    return post


# ============================================
# TRENDING ENDPOINTS
# ============================================

@app.get("/trending", response_model=List[TrendingPost])
async def get_trending_posts(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get trending posts with scores and topics"""
    query = select(Post, TrendScore, Topic).join(
        TrendScore, Post.id == TrendScore.post_id
    ).outerjoin(
        Topic, Post.topic_id == Topic.id
    ).where(
        Post.is_trending == True
    ).order_by(
        desc(TrendScore.total_score)
    ).limit(limit)
    
    result = await db.execute(query)
    rows = result.all()
    
    trending_posts = []
    for post, trend_score, topic in rows:
        trending_posts.append({
            "post": PostResponse.model_validate(post),
            "trend_score": TrendScoreResponse.model_validate(trend_score),
            "topic": TopicResponse.model_validate(topic) if topic else None
        })
    
    return trending_posts


# ============================================
# TOPIC ENDPOINTS
# ============================================

@app.get("/topics", response_model=List[TopicResponse])
async def get_topics(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Get all topics"""
    query = select(Topic).order_by(desc(Topic.num_posts)).offset(skip).limit(limit)
    result = await db.execute(query)
    topics = result.scalars().all()
    
    return topics


@app.get("/topics/{topic_id}", response_model=TopicWithPosts)
async def get_topic(
    topic_id: int,
    include_posts: bool = Query(True),
    post_limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get topic with associated posts"""
    result = await db.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalar_one_or_none()
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    response = TopicWithPosts.model_validate(topic)
    
    if include_posts:
        posts_query = select(Post).where(
            Post.topic_id == topic_id
        ).order_by(desc(Post.score)).limit(post_limit)
        
        posts_result = await db.execute(posts_query)
        response.posts = [PostResponse.model_validate(p) for p in posts_result.scalars().all()]
    
    return response


# ============================================
# SENTIMENT ANALYTICS
# ============================================

@app.get("/sentiment/stats", response_model=SentimentStats)
async def get_sentiment_stats(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: AsyncSession = Depends(get_db)
):
    """Get sentiment distribution statistics"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    # Count by sentiment
    positive_count = await db.scalar(
        select(func.count(Post.id)).where(
            Post.sentiment_label == 'positive',
            Post.created_at >= cutoff_time
        )
    )
    
    negative_count = await db.scalar(
        select(func.count(Post.id)).where(
            Post.sentiment_label == 'negative',
            Post.created_at >= cutoff_time
        )
    )
    
    neutral_count = await db.scalar(
        select(func.count(Post.id)).where(
            Post.sentiment_label == 'neutral',
            Post.created_at >= cutoff_time
        )
    )
    
    # Average sentiment
    avg_sentiment = await db.scalar(
        select(func.avg(Post.sentiment_score)).where(
            Post.created_at >= cutoff_time
        )
    ) or 0.0
    
    # Determine trend
    sentiment_trend = "stable"
    if avg_sentiment > 0.1:
        sentiment_trend = "increasing"
    elif avg_sentiment < -0.1:
        sentiment_trend = "decreasing"
    
    return {
        "positive_count": positive_count or 0,
        "negative_count": negative_count or 0,
        "neutral_count": neutral_count or 0,
        "avg_sentiment": float(avg_sentiment),
        "sentiment_trend": sentiment_trend
    }


# ============================================
# DASHBOARD STATS
# ============================================

@app.get("/stats/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """Get comprehensive dashboard statistics"""
    # Total counts
    total_posts = await db.scalar(select(func.count(Post.id)))
    total_topics = await db.scalar(select(func.count(Topic.id)))
    trending_count = await db.scalar(
        select(func.count(Post.id)).where(Post.is_trending == True)
    )
    
    # Sentiment stats
    sentiment_stats = await get_sentiment_stats(hours=24, db=db)
    
    # Top topics
    top_topics_query = select(Topic).order_by(desc(Topic.num_posts)).limit(5)
    result = await db.execute(top_topics_query)
    top_topics_data = result.scalars().all()
    
    top_topics = [
        TopicStats(
            topic_id=t.id,
            topic_name=t.name,
            post_count=t.num_posts,
            avg_sentiment=t.avg_sentiment or 0.0,
            growth_rate=0.0  # TODO: Calculate from topic_drift
        )
        for t in top_topics_data
    ]
    
    return {
        "total_posts": total_posts or 0,
        "total_topics": total_topics or 0,
        "trending_posts": trending_count or 0,
        "sentiment_distribution": sentiment_stats,
        "top_topics": top_topics,
        "last_updated": datetime.utcnow()
    }


# ============================================
# CHATBOT ENDPOINT (Placeholder)
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Chatbot endpoint - connects to LLM and FAISS search"""
    # TODO: Implement in app/chatbot.py
    return {
        "response": "Chatbot endpoint - implementation coming in chatbot.py",
        "relevant_posts": [],
        "sources": [],
        "confidence": 0.0
    }


# ============================================
# SCRAPER CONTROL (Placeholder)
# ============================================

@app.post("/scrape", response_model=ScraperStatus)
async def trigger_scrape(config: ScraperConfig):
    """Trigger a scraping job"""
    # TODO: Implement async job queue
    return {
        "status": "running",
        "posts_scraped": 0,
        "start_time": datetime.utcnow(),
        "error": None
    }
