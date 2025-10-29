"""
Topic Drift Detection
Tracks how topics evolve over time by comparing snapshots
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from sqlalchemy import select, desc

from backend.db import DatabaseSession
from backend.models import Topic, TopicDrift, Post
from pipeline.topic_model import topic_manager

# ============================================
# SIMILARITY CALCULATIONS
# ============================================

def calculate_keyword_similarity(
    keywords1: List[str],
    keywords2: List[str]
) -> float:
    """
    Calculate similarity between two keyword lists
    Uses Jaccard similarity (intersection / union)
    
    Args:
        keywords1: First keyword list
        keywords2: Second keyword list
    
    Returns:
        Similarity score 0-1
    """
    if not keywords1 or not keywords2:
        return 0.0
    
    set1 = set(keywords1)
    set2 = set(keywords2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_sentiment_change(
    current_sentiment: float,
    previous_sentiment: float
) -> float:
    """
    Calculate sentiment change between snapshots
    
    Args:
        current_sentiment: Recent sentiment (-1 to 1)
        previous_sentiment: Previous sentiment (-1 to 1)
    
    Returns:
        Change value (negative = decreased, positive = increased)
    """
    return current_sentiment - previous_sentiment


def detect_drift_level(
    keyword_similarity: float,
    sentiment_change: float,
    post_count_change: float
) -> str:
    """
    Classify drift level based on metrics
    
    Args:
        keyword_similarity: 0-1 (1 = identical keywords)
        sentiment_change: Sentiment delta
        post_count_change: Relative change in post count
    
    Returns:
        Drift level: 'stable', 'minor', 'moderate', 'major'
    """
    # High similarity = stable
    if keyword_similarity > 0.7 and abs(sentiment_change) < 0.2:
        return 'stable'
    
    # Low similarity or big changes = major drift
    if keyword_similarity < 0.3 or abs(sentiment_change) > 0.5:
        return 'major'
    
    # Medium changes = moderate drift
    if keyword_similarity < 0.5 or abs(sentiment_change) > 0.3:
        return 'moderate'
    
    return 'minor'


# ============================================
# SNAPSHOT CREATION
# ============================================

async def create_topic_snapshot(
    topic_id: int,
    time_window_hours: int = 24
) -> Dict:
    """
    Create a snapshot of current topic state
    
    Args:
        topic_id: Topic database ID
        time_window_hours: Look at posts from last N hours
    
    Returns:
        Snapshot dictionary
    """
    async with DatabaseSession() as db:
        # Get topic
        topic = await db.scalar(select(Topic).where(Topic.id == topic_id))
        
        if not topic:
            return None
        
        # Get recent posts for this topic
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        posts_result = await db.execute(
            select(Post).where(
                Post.topic_id == topic_id,
                Post.created_at >= cutoff_time
            ).order_by(desc(Post.created_at))
        )
        recent_posts = posts_result.scalars().all()
        
        if not recent_posts:
            return None
        
        # Calculate metrics
        sentiments = [p.sentiment_score for p in recent_posts if p.sentiment_score is not None]
        avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
        
        top_post_ids = [p.id for p in recent_posts[:10]]
        
        snapshot = {
            'topic_id': topic_id,
            'snapshot_time': datetime.utcnow(),
            'keywords': topic.keywords,
            'post_count': len(recent_posts),
            'avg_sentiment': avg_sentiment,
            'top_posts': top_post_ids
        }
        
        return snapshot


async def save_snapshot(snapshot: Dict):
    """Save snapshot to database"""
    async with DatabaseSession() as db:
        drift = TopicDrift(
            topic_id=snapshot['topic_id'],
            snapshot_time=snapshot['snapshot_time'],
            keywords=snapshot['keywords'],
            post_count=snapshot['post_count'],
            avg_sentiment=snapshot['avg_sentiment'],
            top_posts=snapshot['top_posts']
        )
        
        db.add(drift)
        await db.commit()


async def create_all_snapshots(time_window_hours: int = 24):
    """
    Create snapshots for all topics
    
    Args:
        time_window_hours: Time window for analysis
    """
    print(f"📸 Creating snapshots (last {time_window_hours} hours)...")
    
    async with DatabaseSession() as db:
        # Get all topics
        topics_result = await db.execute(select(Topic))
        topics = topics_result.scalars().all()
        
        print(f"   Found {len(topics)} topics")
        
        created = 0
        
        for topic in topics:
            snapshot = await create_topic_snapshot(topic.id, time_window_hours)
            
            if snapshot:
                await save_snapshot(snapshot)
                created += 1
        
        print(f"✅ Created {created} snapshots")


# ============================================
# DRIFT ANALYSIS
# ============================================

async def analyze_topic_drift(
    topic_id: int,
    compare_hours: int = 24
) -> Optional[Dict]:
    """
    Analyze drift for a topic by comparing with previous snapshot
    
    Args:
        topic_id: Topic to analyze
        compare_hours: Compare with snapshot from N hours ago
    
    Returns:
        Drift analysis dictionary
    """
    async with DatabaseSession() as db:
        # Get topic
        topic = await db.scalar(select(Topic).where(Topic.id == topic_id))
        
        if not topic:
            return None
        
        # Get two most recent snapshots
        snapshots_result = await db.execute(
            select(TopicDrift).where(
                TopicDrift.topic_id == topic_id
            ).order_by(desc(TopicDrift.snapshot_time)).limit(2)
        )
        snapshots = snapshots_result.scalars().all()
        
        if len(snapshots) < 2:
            return {
                'topic_id': topic_id,
                'topic_name': topic.name,
                'status': 'insufficient_data',
                'message': 'Need at least 2 snapshots for comparison'
            }
        
        current = snapshots[0]
        previous = snapshots[1]
        
        # Calculate metrics
        keyword_sim = calculate_keyword_similarity(
            current.keywords,
            previous.keywords
        )
        
        sentiment_delta = calculate_sentiment_change(
            current.avg_sentiment,
            previous.avg_sentiment
        )
        
        post_count_change = (
            (current.post_count - previous.post_count) / max(previous.post_count, 1)
        )
        
        drift_level = detect_drift_level(
            keyword_sim,
            sentiment_delta,
            post_count_change
        )
        
        # Update current snapshot with drift metrics
        current.keyword_similarity = keyword_sim
        current.sentiment_change = sentiment_delta
        await db.commit()
        
        return {
            'topic_id': topic_id,
            'topic_name': topic.name,
            'drift_level': drift_level,
            'keyword_similarity': keyword_sim,
            'sentiment_change': sentiment_delta,
            'post_count_change': post_count_change,
            'current_snapshot_time': current.snapshot_time,
            'previous_snapshot_time': previous.snapshot_time,
            'keywords_added': list(set(current.keywords) - set(previous.keywords)),
            'keywords_removed': list(set(previous.keywords) - set(current.keywords))
        }


async def analyze_all_drifts() -> List[Dict]:
    """
    Analyze drift for all topics
    
    Returns:
        List of drift analysis results
    """
    print("🔍 Analyzing topic drift...")
    
    async with DatabaseSession() as db:
        topics_result = await db.execute(select(Topic))
        topics = topics_result.scalars().all()
        
        results = []
        
        for topic in topics:
            drift = await analyze_topic_drift(topic.id)
            if drift and drift.get('status') != 'insufficient_data':
                results.append(drift)
        
        # Sort by drift level
        drift_order = {'major': 0, 'moderate': 1, 'minor': 2, 'stable': 3}
        results.sort(key=lambda x: drift_order.get(x['drift_level'], 4))
        
        print(f"✅ Analyzed {len(results)} topics")
        
        return results


# ============================================
# TRENDING DRIFT DETECTION
# ============================================

async def detect_emerging_topics() -> List[Dict]:
    """
    Detect topics that are rapidly growing (emerging trends)
    
    Returns:
        List of emerging topics with growth metrics
    """
    print("🌱 Detecting emerging topics...")
    
    async with DatabaseSession() as db:
        # Get all topics with recent activity
        topics_result = await db.execute(
            select(Topic).where(Topic.num_posts > 5)
        )
        topics = topics_result.scalars().all()
        
        emerging = []
        
        for topic in topics:
            # Get last 2 snapshots
            snapshots_result = await db.execute(
                select(TopicDrift).where(
                    TopicDrift.topic_id == topic.id
                ).order_by(desc(TopicDrift.snapshot_time)).limit(2)
            )
            snapshots = snapshots_result.scalars().all()
            
            if len(snapshots) < 2:
                continue
            
            current = snapshots[0]
            previous = snapshots[1]
            
            # Calculate growth rate
            if previous.post_count > 0:
                growth_rate = (current.post_count - previous.post_count) / previous.post_count
            else:
                growth_rate = 1.0 if current.post_count > 0 else 0.0
            
            # Emerging if growing rapidly
            if growth_rate > 0.5:  # 50% growth
                emerging.append({
                    'topic_id': topic.id,
                    'topic_name': topic.name,
                    'keywords': topic.keywords[:5],
                    'growth_rate': growth_rate,
                    'current_posts': current.post_count,
                    'previous_posts': previous.post_count,
                    'sentiment': current.avg_sentiment
                })
        
        # Sort by growth rate
        emerging.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        print(f"✅ Found {len(emerging)} emerging topics")
        
        return emerging


async def detect_declining_topics() -> List[Dict]:
    """
    Detect topics that are losing interest
    
    Returns:
        List of declining topics
    """
    print("📉 Detecting declining topics...")
    
    async with DatabaseSession() as db:
        topics_result = await db.execute(select(Topic))
        topics = topics_result.scalars().all()
        
        declining = []
        
        for topic in topics:
            snapshots_result = await db.execute(
                select(TopicDrift).where(
                    TopicDrift.topic_id == topic.id
                ).order_by(desc(TopicDrift.snapshot_time)).limit(2)
            )
            snapshots = snapshots_result.scalars().all()
            
            if len(snapshots) < 2:
                continue
            
            current = snapshots[0]
            previous = snapshots[1]
            
            # Calculate decline
            if previous.post_count > 0:
                decline_rate = (previous.post_count - current.post_count) / previous.post_count
            else:
                decline_rate = 0.0
            
            # Declining if losing activity
            if decline_rate > 0.3:  # 30% decline
                declining.append({
                    'topic_id': topic.id,
                    'topic_name': topic.name,
                    'decline_rate': decline_rate,
                    'current_posts': current.post_count,
                    'previous_posts': previous.post_count
                })
        
        declining.sort(key=lambda x: x['decline_rate'], reverse=True)
        
        print(f"✅ Found {len(declining)} declining topics")
        
        return declining


# ============================================
# REPORTING
# ============================================

def print_drift_report(drifts: List[Dict]):
    """Print formatted drift report"""
    print("\n" + "="*60)
    print("📊 TOPIC DRIFT REPORT")
    print("="*60)
    
    for drift in drifts[:10]:  # Top 10
        emoji = {
            'major': '🔴',
            'moderate': '🟡',
            'minor': '🟢',
            'stable': '⚪'
        }.get(drift['drift_level'], '⚫')
        
        print(f"\n{emoji} {drift['topic_name']}")
        print(f"   Drift Level: {drift['drift_level'].upper()}")
        print(f"   Keyword Similarity: {drift['keyword_similarity']:.2%}")
        print(f"   Sentiment Change: {drift['sentiment_change']:+.3f}")
        print(f"   Post Count Change: {drift['post_count_change']:+.1%}")
        
        if drift['keywords_added']:
            print(f"   ➕ New Keywords: {', '.join(drift['keywords_added'][:3])}")
        if drift['keywords_removed']:
            print(f"   ➖ Lost Keywords: {', '.join(drift['keywords_removed'][:3])}")


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Full drift tracking pipeline"""
    print("🚀 Topic Drift Tracking Pipeline\n")
    
    # Step 1: Create snapshots
    await create_all_snapshots(time_window_hours=24)
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: Analyze drift
    drifts = await analyze_all_drifts()
    print_drift_report(drifts)
    
    print("\n" + "="*60 + "\n")
    
    # Step 3: Detect emerging topics
    emerging = await detect_emerging_topics()
    
    if emerging:
        print("\n🌱 EMERGING TOPICS:")
        for topic in emerging[:5]:
            print(f"\n   📈 {topic['topic_name']}")
            print(f"      Growth: {topic['growth_rate']:+.1%}")
            print(f"      Posts: {topic['previous_posts']} → {topic['current_posts']}")
            print(f"      Keywords: {', '.join(topic['keywords'])}")
    
    print("\n" + "="*60 + "\n")
    
    # Step 4: Detect declining topics
    declining = await detect_declining_topics()
    
    if declining:
        print("\n📉 DECLINING TOPICS:")
        for topic in declining[:5]:
            print(f"\n   📉 {topic['topic_name']}")
            print(f"      Decline: -{topic['decline_rate']:.1%}")
            print(f"      Posts: {topic['previous_posts']} → {topic['current_posts']}")
    
    print("\n✅ Drift tracking complete!")


if __name__ == "__main__":
    asyncio.run(main())
