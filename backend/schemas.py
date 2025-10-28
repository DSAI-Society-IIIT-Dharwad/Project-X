"""
Pydantic Schemas for API Request/Response Validation
Separates API layer from database models
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


# ============================================
# POST SCHEMAS
# ============================================

class PostBase(BaseModel):
    """Base schema for Post - shared fields"""
    title: str = Field(..., min_length=1, max_length=500)
    content: Optional[str] = None
    url: Optional[str] = None
    subreddit: Optional[str] = None
    author: Optional[str] = None


class PostCreate(PostBase):
    """Schema for creating a new post"""
    post_id: str = Field(..., description="Unique ID from Reddit/Twitter")
    source: str = Field(..., description="'reddit' or 'twitter'")
    score: int = Field(default=0)
    num_comments: int = Field(default=0)
    created_at: datetime


class PostUpdate(BaseModel):
    """Schema for updating existing post"""
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_label: Optional[str] = None
    topic_id: Optional[int] = None
    is_processed: Optional[bool] = None
    is_trending: Optional[bool] = None


class PostResponse(PostBase):
    """Schema for post in API responses"""
    id: int
    post_id: str
    source: str
    score: int
    num_comments: int
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    topic_id: Optional[int] = None
    is_trending: bool
    created_at: datetime
    scraped_at: datetime
    
    model_config = ConfigDict(from_attributes=True)  # Pydantic v2 style


class PostWithTopic(PostResponse):
    """Post response with topic details included"""
    topic_name: Optional[str] = None
    topic_keywords: Optional[List[str]] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# TOPIC SCHEMAS
# ============================================

class TopicBase(BaseModel):
    """Base schema for Topic"""
    name: str = Field(..., max_length=200)
    keywords: List[str] = Field(default_factory=list)


class TopicCreate(TopicBase):
    """Schema for creating new topic"""
    topic_num: int = Field(..., description="BERTopic topic number")


class TopicUpdate(BaseModel):
    """Schema for updating topic"""
    name: Optional[str] = None
    keywords: Optional[List[str]] = None
    num_posts: Optional[int] = None
    avg_sentiment: Optional[float] = None


class TopicResponse(TopicBase):
    """Schema for topic in API responses"""
    id: int
    topic_num: int
    num_posts: int
    avg_sentiment: Optional[float] = None
    first_seen: datetime
    last_updated: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class TopicWithPosts(TopicResponse):
    """Topic with associated posts"""
    posts: List[PostResponse] = Field(default_factory=list)
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# TREND SCORE SCHEMAS
# ============================================

class TrendScoreBase(BaseModel):
    """Base schema for TrendScore"""
    frequency_score: float = Field(default=0.0)
    velocity_score: float = Field(default=0.0)
    sentiment_change: float = Field(default=0.0)
    total_score: float = Field(..., description="Combined trend score")


class TrendScoreCreate(TrendScoreBase):
    """Schema for creating trend score"""
    post_id: int


class TrendScoreResponse(TrendScoreBase):
    """Schema for trend score in API responses"""
    id: int
    post_id: int
    rank: Optional[int] = None
    calculated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class TrendingPost(BaseModel):
    """Combined schema for trending posts with all info"""
    post: PostResponse
    trend_score: TrendScoreResponse
    topic: Optional[TopicResponse] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# TOPIC DRIFT SCHEMAS
# ============================================

class TopicDriftCreate(BaseModel):
    """Schema for creating topic drift snapshot"""
    topic_id: int
    snapshot_time: datetime
    keywords: List[str]
    post_count: int
    avg_sentiment: float
    top_posts: List[int] = Field(default_factory=list, description="List of post IDs")


class TopicDriftResponse(BaseModel):
    """Schema for topic drift in API responses"""
    id: int
    topic_id: int
    snapshot_time: datetime
    keywords: List[str]
    post_count: int
    avg_sentiment: float
    keyword_similarity: Optional[float] = None
    sentiment_change: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# SUMMARY SCHEMAS
# ============================================

class SummaryRequest(BaseModel):
    """Request schema for generating summaries"""
    summary_type: str = Field(..., description="'topic', 'trending', or 'daily'")
    topic_id: Optional[int] = None
    date: Optional[datetime] = None


class SummaryResponse(BaseModel):
    """Schema for summary in API responses"""
    id: int
    cache_key: str
    summary_type: str
    summary_text: str
    generated_at: datetime
    token_count: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# ANALYTICS SCHEMAS
# ============================================

class SentimentStats(BaseModel):
    """Schema for sentiment statistics"""
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_sentiment: float
    sentiment_trend: str = Field(..., description="'increasing', 'decreasing', or 'stable'")


class TopicStats(BaseModel):
    """Schema for topic statistics"""
    topic_id: int
    topic_name: str
    post_count: int
    avg_sentiment: float
    growth_rate: float = Field(..., description="Percentage change in posts")


class DashboardStats(BaseModel):
    """Schema for dashboard overview"""
    total_posts: int
    total_topics: int
    trending_posts: int
    sentiment_distribution: SentimentStats
    top_topics: List[TopicStats]
    last_updated: datetime


# ============================================
# CHATBOT SCHEMAS
# ============================================

class ChatRequest(BaseModel):
    """Request schema for chatbot"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response schema for chatbot"""
    response: str
    relevant_posts: List[PostResponse] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list, description="URLs to sources")
    confidence: float = Field(..., ge=0.0, le=1.0)


# ============================================
# SCRAPER SCHEMAS
# ============================================

class ScraperConfig(BaseModel):
    """Configuration for scraping job"""
    subreddits: List[str] = Field(default_factory=list)
    keywords: Optional[List[str]] = None
    limit: int = Field(default=100, ge=1, le=1000)
    time_filter: str = Field(default="day", description="'hour', 'day', 'week', 'month'")


class ScraperStatus(BaseModel):
    """Status of scraping job"""
    status: str = Field(..., description="'running', 'completed', 'failed'")
    posts_scraped: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None
