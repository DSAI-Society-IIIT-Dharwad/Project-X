"""
Streamlit Dashboard for News Bot
Interactive web interface for exploring trends, topics, and sentiment
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc
import sys

from backend.db import DatabaseSession
from backend.models import Post, Topic, TrendScore
from pipeline.summary import generate_trending_summary, generate_topic_summary

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="NewsBot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .trend-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.875rem;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
    }
    .neutral {
        background-color: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_overview_stats():
    """Get dashboard overview statistics"""
    async def fetch_stats():
        async with DatabaseSession() as db:
            total_posts = await db.scalar(select(func.count(Post.id)))
            total_topics = await db.scalar(select(func.count(Topic.id)))
            trending_count = await db.scalar(
                select(func.count(Post.id)).where(Post.is_trending == True)
            )
            
            # Sentiment distribution
            positive = await db.scalar(
                select(func.count(Post.id)).where(Post.sentiment_label == 'positive')
            )
            negative = await db.scalar(
                select(func.count(Post.id)).where(Post.sentiment_label == 'negative')
            )
            neutral = await db.scalar(
                select(func.count(Post.id)).where(Post.sentiment_label == 'neutral')
            )
            
            # Average sentiment
            avg_sentiment = await db.scalar(select(func.avg(Post.sentiment_score)))
            
            return {
                'total_posts': total_posts or 0,
                'total_topics': total_topics or 0,
                'trending_count': trending_count or 0,
                'positive': positive or 0,
                'negative': negative or 0,
                'neutral': neutral or 0,
                'avg_sentiment': float(avg_sentiment) if avg_sentiment else 0.0
            }
    
    return run_async(fetch_stats())

@st.cache_data(ttl=300)
def get_top_topics(limit=10):
    """Get top topics by post count"""
    async def fetch_topics():
        async with DatabaseSession() as db:
            result = await db.execute(
                select(Topic).order_by(desc(Topic.num_posts)).limit(limit)
            )
            topics = result.scalars().all()
            
            return [
                {
                    'id': t.id,
                    'name': t.name,
                    'keywords': ', '.join(t.keywords[:5]) if t.keywords else '',
                    'num_posts': t.num_posts,
                    'avg_sentiment': t.avg_sentiment or 0.0
                }
                for t in topics
            ]
    
    return run_async(fetch_topics())

@st.cache_data(ttl=300)
def get_trending_posts(limit=20):
    """Get trending posts"""
    async def fetch_trending():
        async with DatabaseSession() as db:
            result = await db.execute(
                select(Post, TrendScore, Topic).join(
                    TrendScore, Post.id == TrendScore.post_id
                ).outerjoin(
                    Topic, Post.topic_id == Topic.id
                ).where(
                    Post.is_trending == True
                ).order_by(desc(TrendScore.total_score)).limit(limit)
            )
            
            rows = result.all()
            
            trending = []
            for post, trend, topic in rows:
                trending.append({
                    'title': post.title,
                    'subreddit': post.subreddit,
                    'score': post.score,
                    'comments': post.num_comments,
                    'sentiment': post.sentiment_label,
                    'sentiment_score': post.sentiment_score or 0.0,
                    'trend_score': trend.total_score,
                    'topic': topic.name if topic else 'Uncategorized',
                    'url': post.url,
                    'created_at': post.created_at
                })
            
            return trending
    
    return run_async(fetch_trending())

@st.cache_data(ttl=300)
def get_posts_over_time(hours=24):
    """Get post count over time"""
    async def fetch_time_data():
        async with DatabaseSession() as db:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            result = await db.execute(
                select(Post).where(Post.created_at >= cutoff).order_by(Post.created_at)
            )
            posts = result.scalars().all()
            
            # Group by hour
            time_data = {}
            for post in posts:
                hour = post.created_at.replace(minute=0, second=0, microsecond=0)
                if hour not in time_data:
                    time_data[hour] = {'count': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
                
                time_data[hour]['count'] += 1
                if post.sentiment_label == 'positive':
                    time_data[hour]['positive'] += 1
                elif post.sentiment_label == 'negative':
                    time_data[hour]['negative'] += 1
                elif post.sentiment_label == 'neutral':
                    time_data[hour]['neutral'] += 1
            
            return sorted(time_data.items())
    
    return run_async(fetch_time_data())

# ============================================
# DASHBOARD COMPONENTS
# ============================================

def render_header():
    """Render dashboard header"""
    st.markdown('<h1 class="main-header">🤖 NewsBot Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

def render_metrics(stats):
    """Render key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📰 Total Posts",
            value=f"{stats['total_posts']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🏷️ Topics",
            value=stats['total_topics'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="🔥 Trending",
            value=stats['trending_count'],
            delta=None
        )
    
    with col4:
        sentiment_emoji = "😊" if stats['avg_sentiment'] > 0.1 else "😐" if stats['avg_sentiment'] > -0.1 else "😟"
        st.metric(
            label=f"{sentiment_emoji} Avg Sentiment",
            value=f"{stats['avg_sentiment']:.3f}",
            delta=None
        )

def render_sentiment_pie(stats):
    """Render sentiment distribution pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[stats['positive'], stats['neutral'], stats['negative']],
        marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
        hole=0.3
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def render_topics_bar(topics_data):
    """Render top topics bar chart"""
    df = pd.DataFrame(topics_data)
    
    fig = px.bar(
        df,
        x='num_posts',
        y='name',
        orientation='h',
        title="Top Topics by Post Count",
        labels={'num_posts': 'Number of Posts', 'name': 'Topic'},
        color='avg_sentiment',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    
    fig.update_layout(height=500)
    
    return fig

def render_timeline(time_data):
    """Render posts over time"""
    if not time_data:
        return None
    
    times = [t[0] for t in time_data]
    counts = [t[1]['count'] for t in time_data]
    positive = [t[1]['positive'] for t in time_data]
    negative = [t[1]['negative'] for t in time_data]
    neutral = [t[1]['neutral'] for t in time_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, y=positive,
        name='Positive',
        stackgroup='one',
        fillcolor='rgba(40, 167, 69, 0.5)'
    ))
    
    fig.add_trace(go.Scatter(
        x=times, y=neutral,
        name='Neutral',
        stackgroup='one',
        fillcolor='rgba(255, 193, 7, 0.5)'
    ))
    
    fig.add_trace(go.Scatter(
        x=times, y=negative,
        name='Negative',
        stackgroup='one',
        fillcolor='rgba(220, 53, 69, 0.5)'
    ))
    
    fig.update_layout(
        title="Posts Over Time (Stacked by Sentiment)",
        xaxis_title="Time",
        yaxis_title="Number of Posts",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def render_trending_table(trending_data):
    """Render trending posts table"""
    if not trending_data:
        st.info("No trending posts found")
        return
    
    for i, post in enumerate(trending_data[:10], 1):
        with st.expander(f"#{i} - {post['title'][:80]}..."):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Topic:** {post['topic']}")
                st.markdown(f"**Subreddit:** r/{post['subreddit']}")
            
            with col2:
                sentiment_class = post['sentiment']
                sentiment_emoji = "😊" if sentiment_class == 'positive' else "😐" if sentiment_class == 'neutral' else "😟"
                st.markdown(f"**Sentiment:** {sentiment_emoji} {sentiment_class.capitalize()}")
                st.markdown(f"**Score:** {post['sentiment_score']:.3f}")
            
            with col3:
                st.markdown(f"**Upvotes:** {post['score']}")
                st.markdown(f"**Comments:** {post['comments']}")
                st.markdown(f"**Trend Score:** {post['trend_score']:.1f}")
            
            if post['url']:
                st.markdown(f"[🔗 View Post]({post['url']})")

# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    """Main dashboard application"""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        time_window = st.selectbox(
            "Time Window",
            options=[6, 12, 24, 48, 168],
            format_func=lambda x: f"Last {x} hours" if x < 168 else "Last week",
            index=2
        )
        
        st.markdown("---")
        
        st.header("📊 Quick Stats")
        stats = get_overview_stats()
        st.metric("Total Posts", f"{stats['total_posts']:,}")
        st.metric("Topics", stats['total_topics'])
        st.metric("Trending Now", stats['trending_count'])
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    stats = get_overview_stats()
    
    # Metrics row
    render_metrics(stats)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_sentiment_pie(stats), use_container_width=True)
    
    with col2:
        topics_data = get_top_topics(limit=10)
        st.plotly_chart(render_topics_bar(topics_data), use_container_width=True)
    
    # Timeline
    st.markdown("---")
    time_data = get_posts_over_time(hours=time_window)
    timeline_fig = render_timeline(time_data)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Trending posts
    st.markdown("---")
    st.header("🔥 Trending Posts")
    trending_data = get_trending_posts(limit=20)
    render_trending_table(trending_data)
    
    # AI Summary (optional)
    st.markdown("---")
    st.header("🤖 AI-Generated Summary")
    
    if st.button("Generate Trending Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary = run_async(generate_trending_summary(use_cache=True))
                st.success(summary)
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")


# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()
