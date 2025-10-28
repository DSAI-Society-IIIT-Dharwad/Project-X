"""
Trend Score Calculation
Identifies trending posts based on frequency, velocity, and sentiment shifts
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import select, func, desc
from collections import Counter

from backend.db import DatabaseSession
from backend.models import Post, Topic, TrendScore

# ============================================
# SCORING COMPONENTS
# ============================================

def calculate_frequency_score(
    post_count: int,
    time_window_hours: int = 24,
    total_posts: int = 1000
) -> float:
    """
    Calculate frequency score based on post volume
    
    Args:
        post_count: Number of posts on this topic
        time_window_hours: Time window for counting
        total_posts: Total posts in system for normalization
    
    Returns:
        Score 0-1
    """
    if total_posts == 0:
        return 0.0
    
    # Normalize by total posts
    frequency = post_count / total_posts
    
    # Apply logarithmic scaling to handle outliers
    frequency_score = np.log1p(frequency * 100) / np.log1p(100)
    
    return min(frequency_score, 1.0)


def calculate_velocity_score(
    recent_count: int,
    previous_count: int,
    decay_factor: float = 0.5
) -> float:
    """
    Calculate velocity score based on growth rate
    
    Args:
        recent_count: Posts in recent time window
        previous_count: Posts in previous time window
        decay_factor: How much to decay past counts
    
    Returns:
        Score 0-1 (can exceed 1 for viral content)
    """
    if previous_count == 0:
        # New topic, high velocity if there are recent posts
        return min(recent_count / 10.0, 1.0)
    
    # Calculate growth rate
    growth_rate = (recent_count - previous_count) / previous_count
    
    # Normalize (0.5 = 50% growth)
    velocity_score = (1 + growth_rate) / 2
    
    # Cap at reasonable bounds
    return np.clip(velocity_score, 0.0, 2.0)


def calculate_sentiment_change_score(
    current_sentiment: float,
    previous_sentiment: float,
    threshold: float = 0.2
) -> float:
    """
    Calculate score based on sentiment shift
    Large shifts indicate emerging topics
    
    Args:
        current_sentiment: Recent average sentiment (-1 to 1)
        previous_sentiment: Previous average sentiment
        threshold: Minimum change to be significant
    
    Returns:
        Score 0-1
    """
    sentiment_change = abs(current_sentiment - previous_sentiment)
    
    if sentiment_change < threshold:
        return 0.0
    
    # Normalize change
    change_score = sentiment_change / 2.0  # Max possible change is 2
    
    return min(change_score, 1.0)


def calculate_engagement_score(
    avg_score: float,
    avg_comments: float,
    max_score: float = 1000.0,
    max_comments: float = 100.0
) -> float:
    """
    Calculate engagement score from upvotes and comments
    
    Args:
        avg_score: Average upvotes/likes
        avg_comments: Average comments
        max_score: Maximum expected score for normalization
        max_comments: Maximum expected comments
    
    Returns:
        Score 0-1
    """
    score_component = min(avg_score / max_score, 1.0) * 0.6
    comment_component = min(avg_comments / max_comments, 1.0) * 0.4
    
    return score_component + comment_component


# ============================================
# TREND SCORE CALCULATION
# ============================================

def calculate_trend_score(
    frequency_score: float,
    velocity_score: float,
    sentiment_change: float,
    engagement_score: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate final trend score as weighted combination
    
    Args:
        frequency_score: How often topic appears
        velocity_score: Rate of increase
        sentiment_change: Shift in sentiment
        engagement_score: User engagement metrics
        weights: Custom weights for components
    
    Returns:
        Final trend score (typically 0-100)
    """
    if weights is None:
        weights = {
            'frequency': 0.25,
            'velocity': 0.35,
            'sentiment': 0.20,
            'engagement': 0.20
        }
    
    # Weighted sum
    score = (
        frequency_score * weights['frequency'] +
        velocity_score * weights['velocity'] +
        sentiment_change * weights['sentiment'] +
        engagement_score * weights['engagement']
    )
    
    # Scale to 0-100
    trend_score = score * 100
    
    return trend_score


# ============================================
# TIME WINDOW ANALYSIS
# ============================================

async def get_posts_in_window(
    db,
    hours_ago: int,
    hours_duration: int,
    topic_id: Optional[int] = None
) -> List[Post]:
    """
    Get posts within a specific time window
    
    Args:
        db: Database session
        hours_ago: How many hours back to start
        hours_duration: Duration of window in hours
        topic_id: Filter by topic (optional)
    
    Returns:
        List of posts
    """
    end_time = datetime.utcnow() - timedelta(hours=hours_ago)
    start_time = end_time - timedelta(hours=hours_duration)
    
    query = select(Post).where(
        Post.created_at >= start_time,
        Post.created_at < end_time
    )
    
    if topic_id:
        query = query.where(Post.topic_id == topic_id)
    
    result = await db.execute(query)
    return result.scalars().all()


async def analyze_topic_trend(
    db,
    topic_id: int,
    recent_hours: int = 6,
    previous_hours: int = 12
) -> Dict:
    """
    Analyze trend for a specific topic
    
    Args:
        db: Database session
        topic_id: Topic to analyze
        recent_hours: Recent time window
        previous_hours: Previous time window for comparison
    
    Returns:
        Dictionary with trend metrics
    """
    # Get recent posts
    recent_posts = await get_posts_in_window(
        db, hours_ago=0, hours_duration=recent_hours, topic_id=topic_id
    )
    
    # Get previous posts
    previous_posts = await get_posts_in_window(
        db, hours_ago=recent_hours, hours_duration=previous_hours, topic_id=topic_id
    )
    
    # Calculate metrics
    recent_count = len(recent_posts)
    previous_count = len(previous_posts)
    
    # Sentiment metrics
    recent_sentiments = [p.sentiment_score for p in recent_posts if p.sentiment_score is not None]
    previous_sentiments = [p.sentiment_score for p in previous_posts if p.sentiment_score is not None]
    
    current_sentiment = np.mean(recent_sentiments) if recent_sentiments else 0.0
    previous_sentiment = np.mean(previous_sentiments) if previous_sentiments else 0.0
    
    # Engagement metrics
    avg_score = np.mean([p.score for p in recent_posts]) if recent_posts else 0.0
    avg_comments = np.mean([p.num_comments for p in recent_posts]) if recent_posts else 0.0
    
    return {
        'recent_count': recent_count,
        'previous_count': previous_count,
        'current_sentiment': float(current_sentiment),
        'previous_sentiment': float(previous_sentiment),
        'avg_score': float(avg_score),
        'avg_comments': float(avg_comments)
    }


# ============================================
# BATCH TREND CALCULATION
# ============================================

async def calculate_trends_for_all_posts(
    recent_hours: int = 6,
    previous_hours: int = 12
):
    """
    Calculate trend scores for all posts
    
    Args:
        recent_hours: Recent time window
        previous_hours: Previous time window
    """
    print("📈 Calculating trend scores...")
    
    async with DatabaseSession() as db:
        # Get all posts from recent period
        recent_posts = await get_posts_in_window(
            db, hours_ago=0, hours_duration=recent_hours
        )
        
        if not recent_posts:
            print("⚠️  No recent posts found")
            return
        
        print(f"📊 Analyzing {len(recent_posts)} recent posts...")
        
        # Get total post count for normalization
        total_posts = await db.scalar(select(func.count(Post.id)))
        
        # Group posts by topic
        topic_posts = {}
        for post in recent_posts:
            if post.topic_id:
                if post.topic_id not in topic_posts:
                    topic_posts[post.topic_id] = []
                topic_posts[post.topic_id].append(post)
        
        print(f"   Found {len(topic_posts)} active topics")
        
        # Analyze each topic
        trend_scores_list = []
        
        for topic_id, posts in topic_posts.items():
            # Get topic analysis
            metrics = await analyze_topic_trend(
                db, topic_id, recent_hours, previous_hours
            )
            
            # Calculate score components
            freq_score = calculate_frequency_score(
                metrics['recent_count'],
                recent_hours,
                total_posts
            )
            
            vel_score = calculate_velocity_score(
                metrics['recent_count'],
                metrics['previous_count']
            )
            
            sent_change = calculate_sentiment_change_score(
                metrics['current_sentiment'],
                metrics['previous_sentiment']
            )
            
            eng_score = calculate_engagement_score(
                metrics['avg_score'],
                metrics['avg_comments']
            )
            
            # Calculate final score
            final_score = calculate_trend_score(
                freq_score, vel_score, sent_change, eng_score
            )
            
            # Store for each post in topic
            for post in posts:
                trend_scores_list.append({
                    'post': post,
                    'frequency_score': freq_score,
                    'velocity_score': vel_score,
                    'sentiment_change': sent_change,
                    'engagement_score': eng_score,
                    'total_score': final_score
                })
        
        # Rank by score
        trend_scores_list.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Assign ranks
        for rank, item in enumerate(trend_scores_list, 1):
            item['rank'] = rank
        
        # Save to database
        print("💾 Saving trend scores...")
        saved_count = 0
        
        for item in trend_scores_list:
            post = item['post']
            
            # Check if trend score exists
            existing = await db.scalar(
                select(TrendScore).where(TrendScore.post_id == post.id)
            )
            
            if existing:
                # Update existing
                existing.frequency_score = item['frequency_score']
                existing.velocity_score = item['velocity_score']
                existing.sentiment_change = item['sentiment_change']
                existing.total_score = item['total_score']
                existing.rank = item['rank']
                existing.calculated_at = datetime.utcnow()
            else:
                # Create new
                trend_score = TrendScore(
                    post_id=post.id,
                    frequency_score=item['frequency_score'],
                    velocity_score=item['velocity_score'],
                    sentiment_change=item['sentiment_change'],
                    total_score=item['total_score'],
                    rank=item['rank']
                )
                db.add(trend_score)
            
            # Mark post as trending if score is high
            post.is_trending = item['total_score'] > 50.0
            
            saved_count += 1
        
        await db.commit()
        
        print(f"✅ Calculated and saved {saved_count} trend scores")
        
        # Show top trending
        print("\n🔥 Top 10 Trending Posts:")
        for i, item in enumerate(trend_scores_list[:10], 1):
            post = item['post']
            score = item['total_score']
            print(f"\n{i}. [{score:.1f}] {post.title[:60]}...")
            print(f"   Frequency: {item['frequency_score']:.3f} | "
                  f"Velocity: {item['velocity_score']:.3f} | "
                  f"Sentiment Δ: {item['sentiment_change']:.3f}")


# ============================================
# TRENDING DETECTION
# ============================================

async def detect_emerging_trends(threshold: float = 60.0) -> List[Dict]:
    """
    Detect emerging trending topics
    
    Args:
        threshold: Minimum trend score to be considered trending
    
    Returns:
        List of trending topic info
    """
    async with DatabaseSession() as db:
        # Get trending posts
        query = select(Post, TrendScore).join(
            TrendScore, Post.id == TrendScore.post_id
        ).where(
            TrendScore.total_score >= threshold
        ).order_by(desc(TrendScore.total_score))
        
        result = await db.execute(query)
        rows = result.all()
        
        # Group by topic
        topic_trends = {}
        
        for post, trend_score in rows:
            if post.topic_id:
                if post.topic_id not in topic_trends:
                    topic_trends[post.topic_id] = {
                        'posts': [],
                        'avg_score': 0,
                        'max_score': 0
                    }
                
                topic_trends[post.topic_id]['posts'].append({
                    'post': post,
                    'score': trend_score.total_score
                })
                topic_trends[post.topic_id]['max_score'] = max(
                    topic_trends[post.topic_id]['max_score'],
                    trend_score.total_score
                )
        
        # Calculate averages
        for topic_id in topic_trends:
            scores = [p['score'] for p in topic_trends[topic_id]['posts']]
            topic_trends[topic_id]['avg_score'] = np.mean(scores)
        
        return topic_trends


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Calculate trend scores for all posts"""
    print("🚀 Starting trend calculation pipeline...\n")
    
    # Calculate trends
    await calculate_trends_for_all_posts(recent_hours=6, previous_hours=12)
    
    # Detect emerging trends
    print("\n🔍 Detecting emerging trends...")
    trends = await detect_emerging_trends(threshold=60.0)
    
    print(f"\n✨ Found {len(trends)} trending topics")
    
    print("\n✅ Trend calculation complete!")


if __name__ == "__main__":
    asyncio.run(main())
