"""
Chat and Query History Storage
Helper functions for storing conversation data
"""

from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import select, desc
from backend.db import DatabaseSession
from backend.models import ChatHistory, QueryAnalysis, DashboardRefresh, UserSession

# ============================================
# CHAT HISTORY
# ============================================

async def save_chat_interaction(
    user_query: str,
    bot_response: str,
    relevant_post_ids: List[int],
    confidence_score: float,
    session_id: Optional[str] = None,
    response_time_ms: int = 0,
    query_type: str = "general"
):
    """Save a chat interaction to database"""
    async with DatabaseSession() as db:
        chat = ChatHistory(
            session_id=session_id,
            user_query=user_query,
            bot_response=bot_response,
            query_type=query_type,
            relevant_post_ids=relevant_post_ids,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms
        )
        
        db.add(chat)
        await db.commit()
        
        return chat.id


async def get_chat_history(
    session_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict]:
    """Retrieve chat history"""
    async with DatabaseSession() as db:
        query = select(ChatHistory).order_by(desc(ChatHistory.created_at))
        
        if session_id:
            query = query.where(ChatHistory.session_id == session_id)
        
        query = query.limit(limit)
        
        result = await db.execute(query)
        chats = result.scalars().all()
        
        return [
            {
                'id': c.id,
                'user_query': c.user_query,
                'bot_response': c.bot_response,
                'confidence': c.confidence_score,
                'created_at': c.created_at,
                'response_time_ms': c.response_time_ms
            }
            for c in chats
        ]


async def add_user_feedback(chat_id: int, rating: int, feedback: Optional[str] = None):
    """Add user feedback to a chat interaction"""
    async with DatabaseSession() as db:
        chat = await db.scalar(select(ChatHistory).where(ChatHistory.id == chat_id))
        
        if chat:
            chat.user_rating = rating
            chat.user_feedback = feedback
            await db.commit()
            return True
        
        return False


# ============================================
# QUERY ANALYSIS
# ============================================

async def save_query_analysis(
    query: str,
    analysis: Dict,
    post_ids: List[int],
    processing_time_ms: int = 0
):
    """Save query analysis results"""
    async with DatabaseSession() as db:
        query_record = QueryAnalysis(
            query=query,
            posts_found=analysis.get('total_posts', 0),
            posts_processed=analysis.get('total_posts', 0),
            sentiment_positive=analysis['sentiment']['positive'],
            sentiment_negative=analysis['sentiment']['negative'],
            sentiment_neutral=analysis['sentiment']['neutral'],
            avg_sentiment=analysis['sentiment']['average_score'],
            top_keywords=analysis['keywords'],
            top_subreddits=analysis['subreddits'],
            ai_summary=analysis.get('summary'),
            post_ids=post_ids,
            processing_time_ms=processing_time_ms
        )
        
        db.add(query_record)
        await db.commit()
        
        return query_record.id


async def get_query_history(limit: int = 20) -> List[Dict]:
    """Get recent query analysis history"""
    async with DatabaseSession() as db:
        result = await db.execute(
            select(QueryAnalysis)
            .order_by(desc(QueryAnalysis.created_at))
            .limit(limit)
        )
        queries = result.scalars().all()
        
        return [
            {
                'id': q.id,
                'query': q.query,
                'posts_found': q.posts_found,
                'avg_sentiment': q.avg_sentiment,
                'top_keywords': q.top_keywords[:5],
                'created_at': q.created_at
            }
            for q in queries
        ]


async def get_popular_queries(days: int = 7, limit: int = 10) -> List[Dict]:
    """Get most popular queries in last N days"""
    from datetime import timedelta
    from sqlalchemy import func
    
    async with DatabaseSession() as db:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(QueryAnalysis.query, func.count(QueryAnalysis.id).label('count'))
            .where(QueryAnalysis.created_at >= cutoff)
            .group_by(QueryAnalysis.query)
            .order_by(desc('count'))
            .limit(limit)
        )
        
        return [{'query': row[0], 'count': row[1]} for row in result.all()]


# ============================================
# DASHBOARD REFRESH
# ============================================

async def start_refresh_record() -> int:
    """Start tracking a dashboard refresh"""
    async with DatabaseSession() as db:
        refresh = DashboardRefresh(
            status="running",
            started_at=datetime.utcnow()
        )
        
        db.add(refresh)
        await db.commit()
        
        return refresh.id


async def update_refresh_record(
    refresh_id: int,
    posts_scraped: int = 0,
    posts_saved: int = 0,
    posts_skipped: int = 0,
    embeddings_generated: bool = False,
    sentiment_analyzed: bool = False,
    topics_updated: bool = False,
    trends_calculated: bool = False,
    status: str = "running",
    error_message: Optional[str] = None
):
    """Update refresh record"""
    async with DatabaseSession() as db:
        refresh = await db.scalar(select(DashboardRefresh).where(DashboardRefresh.id == refresh_id))
        
        if refresh:
            refresh.posts_scraped = posts_scraped
            refresh.posts_saved = posts_saved
            refresh.posts_skipped = posts_skipped
            refresh.embeddings_generated = embeddings_generated
            refresh.sentiment_analyzed = sentiment_analyzed
            refresh.topics_updated = topics_updated
            refresh.trends_calculated = trends_calculated
            refresh.status = status
            refresh.error_message = error_message
            
            if status in ["completed", "failed"]:
                refresh.completed_at = datetime.utcnow()
                refresh.duration_seconds = int((refresh.completed_at - refresh.started_at).total_seconds())
            
            await db.commit()


async def get_refresh_history(limit: int = 10) -> List[Dict]:
    """Get dashboard refresh history"""
    async with DatabaseSession() as db:
        result = await db.execute(
            select(DashboardRefresh)
            .order_by(desc(DashboardRefresh.started_at))
            .limit(limit)
        )
        refreshes = result.scalars().all()
        
        return [
            {
                'id': r.id,
                'posts_scraped': r.posts_scraped,
                'posts_saved': r.posts_saved,
                'status': r.status,
                'started_at': r.started_at,
                'duration_seconds': r.duration_seconds
            }
            for r in refreshes
        ]


# ============================================
# USER SESSION
# ============================================

async def create_or_get_session(session_id: str) -> int:
    """Create or retrieve user session"""
    async with DatabaseSession() as db:
        session = await db.scalar(
            select(UserSession).where(UserSession.session_id == session_id)
        )
        
        if not session:
            session = UserSession(session_id=session_id)
            db.add(session)
            await db.commit()
        
        return session.id


async def update_session_activity(session_id: str, activity_type: str = "query"):
    """Update session activity"""
    async with DatabaseSession() as db:
        session = await db.scalar(
            select(UserSession).where(UserSession.session_id == session_id)
        )
        
        if session:
            session.last_activity = datetime.utcnow()
            
            if activity_type == "query":
                session.queries_count += 1
            elif activity_type == "refresh":
                session.refreshes_count += 1
            
            await db.commit()
