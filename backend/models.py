"""
Database Models for News Bot
Defines tables for Posts, Topics, Sentiment, and TrendScores
"""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Post(Base):
    """
    Stores scraped posts from Reddit/Twitter with metadata
    """
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Source info
    post_id = Column(String(100), unique=True, index=True, nullable=False)  # Reddit/Twitter ID
    source = Column(String(20), nullable=False, index=True)  # 'reddit' or 'twitter'
    subreddit = Column(String(100), index=True)  # For Reddit posts
    author = Column(String(100), index=True)
    
    # Content
    title = Column(Text, nullable=False)
    content = Column(Text)  # Post body/tweet text
    url = Column(String(500))
    
    # Metrics
    score = Column(Integer, default=0)  # Upvotes/likes
    num_comments = Column(Integer, default=0)
    
    # NLP results (stored as relationships)
    sentiment_score = Column(Float, index=True)  # -1 to 1
    sentiment_label = Column(String(20), index=True)  # 'positive', 'negative', 'neutral'
    
    # Topic assigned by BERTopic
    topic_id = Column(Integer, ForeignKey("topics.id"), index=True)
    topic = relationship("Topic", back_populates="posts")
    
    # Embedding for FAISS search
    embedding = Column(JSON)  # Store as list for retrieval
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False)  # Original post time
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())  # When we scraped it
    processed_at = Column(DateTime(timezone=True), onupdate=func.now())  # When NLP was done
    
    # Flags
    is_processed = Column(Boolean, default=False, index=True)
    is_trending = Column(Boolean, default=False, index=True)
    
    # Relationships
    trend_score_rel = relationship("TrendScore", back_populates="post", uselist=False)

    def __repr__(self):
        return f"<Post(id={self.id}, title={self.title[:30]}..., sentiment={self.sentiment_label})>"


class Topic(Base):
    """
    Stores topics discovered by BERTopic
    """
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    
    # Topic info
    topic_num = Column(Integer, unique=True, index=True)  # BERTopic topic number
    name = Column(String(200), index=True)  # Generated topic name
    keywords = Column(JSON)  # List of top keywords
    
    # Metadata
    num_posts = Column(Integer, default=0)
    avg_sentiment = Column(Float)
    
    # Timestamps
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    posts = relationship("Post", back_populates="topic")
    drift_history = relationship("TopicDrift", back_populates="topic")

    def __repr__(self):
        return f"<Topic(id={self.id}, name={self.name}, posts={self.num_posts})>"


class TrendScore(Base):
    """
    Stores calculated trend scores for posts
    Combines frequency, velocity, and sentiment metrics
    """
    __tablename__ = "trend_scores"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    post_id = Column(Integer, ForeignKey("posts.id"), unique=True, nullable=False)
    post = relationship("Post", back_populates="trend_score_rel")
    
    # Score components
    frequency_score = Column(Float, default=0.0)  # How often topic appears
    velocity_score = Column(Float, default=0.0)  # Rate of increase
    sentiment_change = Column(Float, default=0.0)  # Sentiment shift over time
    
    # Final score (weighted combination)
    total_score = Column(Float, index=True, nullable=False)
    
    # Ranking
    rank = Column(Integer, index=True)
    
    # Timestamp
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<TrendScore(post_id={self.post_id}, score={self.total_score}, rank={self.rank})>"


class TopicDrift(Base):
    """
    Tracks how topics evolve over time
    Stores snapshots of topic characteristics
    """
    __tablename__ = "topic_drift"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False, index=True)
    topic = relationship("Topic", back_populates="drift_history")
    
    # Snapshot data
    snapshot_time = Column(DateTime(timezone=True), nullable=False, index=True)
    keywords = Column(JSON)  # Keywords at this time
    post_count = Column(Integer)
    avg_sentiment = Column(Float)
    top_posts = Column(JSON)  # List of post IDs
    
    # Drift metrics
    keyword_similarity = Column(Float)  # Compared to previous snapshot
    sentiment_change = Column(Float)  # Change from previous
    
    def __repr__(self):
        return f"<TopicDrift(topic_id={self.topic_id}, time={self.snapshot_time})>"


class SummaryCache(Base):
    """
    Caches LLM-generated summaries to avoid regenerating
    """
    __tablename__ = "summary_cache"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # What was summarized
    cache_key = Column(String(200), unique=True, index=True, nullable=False)  # e.g., "topic_5_2025-10-29"
    summary_type = Column(String(50), index=True)  # 'topic', 'trending', 'daily'
    
    # The summary
    summary_text = Column(Text, nullable=False)
    
    # Metadata
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))  # For cache invalidation
    token_count = Column(Integer)  # Track API usage
    
    def __repr__(self):
        return f"<SummaryCache(type={self.summary_type}, key={self.cache_key})>"

# ============================================
# ADD THESE NEW MODELS TO backend/models.py
# (Add after the existing models, before the end of file)
# ============================================

class ChatHistory(Base):
    """
    Store chatbot conversation history
    """
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=True)  # For multi-turn conversations
    user_query = Column(Text, nullable=False)  # What user asked
    bot_response = Column(Text, nullable=False)  # Bot's answer
    query_type = Column(String, default="general")  # 'general', 'query_analysis', 'trending'
    
    # Context used
    relevant_post_ids = Column(JSON, default=list)  # List of post IDs used
    confidence_score = Column(Float, default=0.0)  # How confident the bot was
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    response_time_ms = Column(Integer, default=0)  # How long it took
    
    # User feedback (optional)
    user_rating = Column(Integer, nullable=True)  # 1-5 stars
    user_feedback = Column(Text, nullable=True)


class QueryAnalysis(Base):
    """
    Store query-based search history (Workflow 2)
    """
    __tablename__ = "query_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False, index=True)
    
    # Results
    posts_found = Column(Integer, default=0)
    posts_processed = Column(Integer, default=0)
    
    # Analysis summary
    sentiment_positive = Column(Integer, default=0)
    sentiment_negative = Column(Integer, default=0)
    sentiment_neutral = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    
    top_keywords = Column(JSON, default=list)  # Top keywords found
    top_subreddits = Column(JSON, default=dict)  # Subreddit distribution
    
    # AI Summary
    ai_summary = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Integer, default=0)
    
    # Related posts (store IDs for later retrieval)
    post_ids = Column(JSON, default=list)


class DashboardRefresh(Base):
    """
    Track dashboard refresh history (Workflow 1)
    """
    __tablename__ = "dashboard_refresh"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Refresh results
    posts_scraped = Column(Integer, default=0)
    posts_saved = Column(Integer, default=0)
    posts_skipped = Column(Integer, default=0)
    
    # Processing steps completed
    embeddings_generated = Column(Boolean, default=False)
    sentiment_analyzed = Column(Boolean, default=False)
    topics_updated = Column(Boolean, default=False)
    trends_calculated = Column(Boolean, default=False)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)
    
    # Status
    status = Column(String, default="running")  # 'running', 'completed', 'failed'
    error_message = Column(Text, nullable=True)


class UserSession(Base):
    """
    Track user sessions for analytics
    """
    __tablename__ = "user_session"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    
    # Session info
    started_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Activity tracking
    queries_count = Column(Integer, default=0)
    refreshes_count = Column(Integer, default=0)
    
    # User preferences (optional)
    preferred_subreddits = Column(JSON, default=list)
    
    # Metadata
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)  # Store hashed for privacy
