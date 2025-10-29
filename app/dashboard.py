"""
Unified News Dashboard
Shows Reddit + Google News side-by-side with sentiment comparison
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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
from scraper.reddit_scraper import workflow_1_refresh
from scraper.google_news_scraper import workflow_google_news, compare_sentiments
from backend.ingest import process_unprocessed_posts
from pipeline.embeddings import build_faiss_index
from pipeline.topic_model import train_topic_model
from pipeline.trend_score import calculate_trends_for_all_posts

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="📊 Unified News Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLING
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .source-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .comparison-alert {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .agree {
        background: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
    }
    .disagree {
        background: #f8d7da;
        color: #721c24;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def run_async(coro):
    """Run async function"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ============================================
# DATA FETCHING
# ============================================

@st.cache_data(ttl=60)
def get_source_stats():
    """Get stats for both sources"""
    async def fetch():
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
    
    return run_async(fetch())

@st.cache_data(ttl=60)
def get_sentiment_by_source(source):
    """Get sentiment distribution for a source"""
    async def fetch():
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
    
    return run_async(fetch())

@st.cache_data(ttl=60)
def get_recent_posts(source, limit=10):
    """Get recent posts from source"""
    async def fetch():
        async with DatabaseSession() as db:
            result = await db.execute(
                select(Post).where(Post.source == source)
                .order_by(desc(Post.created_at))
                .limit(limit)
            )
            posts = result.scalars().all()
            
            return [
                {
                    'title': p.title,
                    'subreddit': p.subreddit or 'N/A',
                    'sentiment': p.sentiment_label or 'neutral',
                    'sentiment_score': p.sentiment_score or 0.0,
                    'score': p.score,
                    'url': p.url,
                    'created_at': p.created_at
                }
                for p in posts
            ]
    
    return run_async(fetch())

# ============================================
# REFRESH PIPELINE
# ============================================

async def run_full_refresh():
    """Refresh both Reddit and Google News"""
    progress = st.progress(0)
    status = st.empty()
    
    # Step 1: Reddit
    status.text("📡 Scraping Reddit...")
    progress.progress(10)
    reddit_result = await workflow_1_refresh()
    
    # Step 2: Google News
    status.text("📰 Scraping Google News...")
    progress.progress(30)
    gnews_result = await workflow_google_news(GOOGLE_NEWS_TOPICS)
    
    # Step 3: Process
    status.text("⚙️ Processing posts...")
    progress.progress(50)
    await process_unprocessed_posts(limit=None)
    
    # Step 4: Build index
    status.text("🔨 Building search index...")
    progress.progress(70)
    await build_faiss_index()
    
    # Step 5: Topics
    status.text("🧠 Analyzing topics...")
    progress.progress(90)
    
    async with DatabaseSession() as db:
        total = await db.scalar(select(func.count(Post.id)))
        if total > 100:
            await train_topic_model(min_posts=50)
    
    progress.progress(100)
    status.text("✅ Refresh complete!")
    
    return {
        'reddit': reddit_result,
        'google_news': gnews_result
    }

# ============================================
# VISUALIZATIONS
# ============================================

def render_comparison_chart(reddit_sentiment, gnews_sentiment):
    """Side-by-side sentiment comparison"""
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=("Reddit Sentiment", "Google News Sentiment")
    )
    
    # Reddit pie
    fig.add_trace(
        go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[
                reddit_sentiment['positive'],
                reddit_sentiment['neutral'],
                reddit_sentiment['negative']
            ],
            marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
            name="Reddit"
        ),
        row=1, col=1
    )
    
    # Google News pie
    fig.add_trace(
        go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[
                gnews_sentiment['positive'],
                gnews_sentiment['neutral'],
                gnews_sentiment['negative']
            ],
            marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
            name="Google News"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig

def render_sentiment_gauge(reddit_avg, gnews_avg):
    """Dual sentiment gauge"""
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Reddit Average", "Google News Average")
    )
    
    # Reddit gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=reddit_avg,
            domain={'x': [0, 0.48], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "#f8d7da"},
                    {'range': [-0.3, 0.3], 'color': "#fff3cd"},
                    {'range': [0.3, 1], 'color': "#d4edda"}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Google News gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=gnews_avg,
            domain={'x': [0.52, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#764ba2"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "#f8d7da"},
                    {'range': [-0.3, 0.3], 'color': "#fff3cd"},
                    {'range': [0.3, 1], 'color': "#d4edda"}
                ]
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=300)
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    """Main unified dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Unified News Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Compare Reddit discussions with Google News coverage**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Refresh button
        if st.button("🔄 REFRESH ALL DATA", type="primary", use_container_width=True):
            st.session_state['refreshing'] = True
            st.rerun()
        
        st.markdown("---")
        
        st.header("📊 Sources")
        st.info(f"""
        **Reddit:**
        {', '.join([f'r/{s}' for s in MONITORED_SUBREDDITS])}
        
        **Google News Topics:**
        {', '.join(GOOGLE_NEWS_TOPICS[:3])}...
        """)
        
        st.markdown("---")
        
        st.header("🔍 Topic Comparison")
        
        topic = st.text_input("Compare sentiment for:", "artificial intelligence")
        
        if st.button("Compare Topic", use_container_width=True):
            with st.spinner("Analyzing..."):
                comparison = run_async(compare_sentiments(topic))
                st.session_state['topic_comparison'] = comparison
        
        if st.session_state.get('topic_comparison'):
            comp = st.session_state['topic_comparison']
            
            sentiment_class = "agree" if comp['agreement'] else "disagree"
            icon = "✅" if comp['agreement'] else "❌"
            
            st.markdown(f'<div class="comparison-alert {sentiment_class}">', unsafe_allow_html=True)
            st.markdown(f"""
            **{icon} Sentiment Analysis: {comp['topic']}**
            
            Reddit: {comp['reddit']['avg']:.3f}
            Google News: {comp['google_news']['avg']:.3f}
            Difference: {comp['difference']:.3f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle refresh
    if st.session_state.get('refreshing', False):
        result = run_async(run_full_refresh())
        st.success(f"""
        ✅ Refresh Complete!
        
        Reddit: {result['reddit']['posts_saved']} posts
        Google News: {result['google_news']['articles_saved']} articles
        """)
        st.session_state['refreshing'] = False
        st.cache_data.clear()
        st.rerun()
    
    # Get data
    stats = get_source_stats()
    reddit_sentiment = get_sentiment_by_source('reddit')
    gnews_sentiment = get_sentiment_by_source('google_news')
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📱 Reddit Posts", f"{stats['reddit']['total']:,}")
        st.caption(f"Last 24h: {stats['reddit']['recent_24h']}")
    
    with col2:
        st.metric("📰 Google News", f"{stats['google_news']['total']:,}")
        st.caption(f"Last 24h: {stats['google_news']['recent_24h']}")
    
    with col3:
        reddit_emoji = "😊" if stats['reddit']['avg_sentiment'] > 0.2 else "😐" if stats['reddit']['avg_sentiment'] > -0.2 else "😟"
        st.metric(f"{reddit_emoji} Reddit Sentiment", f"{stats['reddit']['avg_sentiment']:.3f}")
    
    with col4:
        gnews_emoji = "😊" if stats['google_news']['avg_sentiment'] > 0.2 else "😐" if stats['google_news']['avg_sentiment'] > -0.2 else "😟"
        st.metric(f"{gnews_emoji} News Sentiment", f"{stats['google_news']['avg_sentiment']:.3f}")
    
    st.markdown("---")
    
    # Sentiment comparison
    st.header("📊 Sentiment Comparison")
    
    # Gauges
    st.plotly_chart(
        render_sentiment_gauge(
            stats['reddit']['avg_sentiment'],
            stats['google_news']['avg_sentiment']
        ),
        use_container_width=True,
        key="gauge_chart"
    )
    
    # Pie charts
    st.plotly_chart(
        render_comparison_chart(reddit_sentiment, gnews_sentiment),
        use_container_width=True,
        key="pie_charts"
    )
    
    # Analysis
    sentiment_diff = abs(stats['reddit']['avg_sentiment'] - stats['google_news']['avg_sentiment'])
    
    if sentiment_diff < 0.2:
        st.success("✅ **Sentiments ALIGN** - Reddit discussions match mainstream news coverage")
    else:
        st.warning(f"⚠️ **Sentiments DIFFER** - {sentiment_diff:.2f} difference between sources")
    
    st.markdown("---")
    
    # Recent posts comparison
    st.header("📋 Recent Posts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Reddit Posts")
        reddit_posts = get_recent_posts('reddit', limit=5)
        
        for i, post in enumerate(reddit_posts, 1):
            sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(post['sentiment'], "😐")
            
            with st.expander(f"{i}. {sentiment_emoji} {post['title'][:50]}..."):
                st.write(f"**r/{post['subreddit']}**")
                st.write(f"Score: {post['score']} | Sentiment: {post['sentiment_score']:.3f}")
                if post['url']:
                    st.markdown(f"[🔗 View on Reddit]({post['url']})")
    
    with col2:
        st.subheader("📰 Google News Articles")
        gnews_posts = get_recent_posts('google_news', limit=5)
        
        for i, post in enumerate(gnews_posts, 1):
            sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(post['sentiment'], "😐")
            
            with st.expander(f"{i}. {sentiment_emoji} {post['title'][:50]}..."):
                st.write(f"**Source: {post['subreddit']}**")
                st.write(f"Sentiment: {post['sentiment_score']:.3f}")
                if post['url']:
                    st.markdown(f"[🔗 Read Article]({post['url']})")

if __name__ == "__main__":
    main()
