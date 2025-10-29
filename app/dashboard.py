"""
Real-Time News Dashboard - Workflow 1
Monitors specific subreddits and updates on refresh
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



from backend.chat_storage import start_refresh_record, update_refresh_record, get_refresh_history
import time
from backend.db import DatabaseSession
from backend.models import Post, Topic
from config import MONITORED_SUBREDDITS
from scraper.reddit_scraper import workflow_1_refresh
from backend.ingest import process_unprocessed_posts
from pipeline.embeddings import build_faiss_index
from pipeline.topic_model import train_topic_model, assign_topics_to_new_posts
from pipeline.trend_score import calculate_trends_for_all_posts
from pipeline.summary import generate_trending_summary

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="📰 Real-Time News Dashboard",
    page_icon="📰",
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
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .refresh-btn {
        font-size: 1.5rem;
        padding: 1rem 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def run_async(coro):
    """Run async function in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ============================================
# DATA FETCHING
# ============================================

@st.cache_data(ttl=60)
def get_dashboard_stats():
    """Get current dashboard statistics"""
    async def fetch():
        async with DatabaseSession() as db:
            total_posts = await db.scalar(select(func.count(Post.id)))
            
            # Recent posts (last 24 hours)
            cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_posts = await db.scalar(
                select(func.count(Post.id)).where(Post.created_at >= cutoff)
            )
            
            # Sentiment stats
            positive = await db.scalar(
                select(func.count(Post.id)).where(Post.sentiment_label == 'positive')
            )
            negative = await db.scalar(
                select(func.count(Post.id)).where(Post.sentiment_label == 'negative')
            )
            neutral = await db.scalar(
                select(func.count(Post.id)).where(Post.sentiment_label == 'neutral')
            )
            
            avg_sentiment = await db.scalar(select(func.avg(Post.sentiment_score)))
            
            # Subreddit distribution
            result = await db.execute(
                select(Post.subreddit, func.count(Post.id))
                .group_by(Post.subreddit)
            )
            subreddit_counts = dict(result.all())
            
            return {
                'total_posts': total_posts or 0,
                'recent_posts': recent_posts or 0,
                'positive': positive or 0,
                'negative': negative or 0,
                'neutral': neutral or 0,
                'avg_sentiment': float(avg_sentiment) if avg_sentiment else 0.0,
                'subreddit_counts': subreddit_counts
            }
    
    return run_async(fetch())

@st.cache_data(ttl=60)
def get_recent_posts(hours=24, limit=50):
    """Get recent posts"""
    async def fetch():
        async with DatabaseSession() as db:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            result = await db.execute(
                select(Post).where(
                    Post.created_at >= cutoff
                ).order_by(desc(Post.created_at)).limit(limit)
            )
            posts = result.scalars().all()
            
            return [
                {
                    'title': p.title,
                    'subreddit': p.subreddit,
                    'score': p.score,
                    'comments': p.num_comments,
                    'sentiment': p.sentiment_label,
                    'sentiment_score': p.sentiment_score or 0.0,
                    'url': p.url,
                    'created_at': p.created_at
                }
                for p in posts
            ]
    
    return run_async(fetch())

@st.cache_data(ttl=60)
def get_posts_timeline(hours=24):
    """Get posts over time for timeline chart"""
    async def fetch():
        async with DatabaseSession() as db:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            result = await db.execute(
                select(Post).where(Post.created_at >= cutoff)
                .order_by(Post.created_at)
            )
            posts = result.scalars().all()
            
            # Group by hour
            timeline = {}
            for post in posts:
                hour = post.created_at.replace(minute=0, second=0, microsecond=0)
                if hour not in timeline:
                    timeline[hour] = {'positive': 0, 'negative': 0, 'neutral': 0}
                
                if post.sentiment_label:
                    timeline[hour][post.sentiment_label] += 1
            
            return timeline
    
    return run_async(fetch())

# ============================================
# REFRESH PIPELINE
# ============================================

async def run_full_refresh_pipeline():
    """
    Complete refresh pipeline with database tracking:
    1. Scrape new posts
    2. Process them
    3. Update analytics
    4. SAVE HISTORY
    """
    # Start tracking
    refresh_id = await start_refresh_record()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Scrape
        status_text.text("📡 Scraping Reddit...")
        progress_bar.progress(10)
        scrape_result = await workflow_1_refresh()
        
        if not scrape_result['success']:
            await update_refresh_record(
                refresh_id, 
                status="failed", 
                error_message="Scraping failed"
            )
            st.error("❌ Scraping failed")
            return False
        
        st.success(f"✅ Scraped {scrape_result['posts_scraped']} posts ({scrape_result['posts_saved']} new)")
        
        await update_refresh_record(
            refresh_id,
            posts_scraped=scrape_result['posts_scraped'],
            posts_saved=scrape_result['posts_saved'],
            posts_skipped=scrape_result['posts_skipped']
        )
        
        # Step 2: Process posts
        status_text.text("⚙️ Processing posts (embeddings, sentiment)...")
        progress_bar.progress(30)
        await process_unprocessed_posts(limit=None)
        
        await update_refresh_record(
            refresh_id,
            embeddings_generated=True,
            sentiment_analyzed=True
        )
        
        # Step 3: Build FAISS index
        status_text.text("🔨 Building search index...")
        progress_bar.progress(50)
        await build_faiss_index()
        
        # Step 4: Train/update topic model
        status_text.text("🧠 Analyzing topics...")
        progress_bar.progress(70)
        
        async with DatabaseSession() as db:
            total = await db.scalar(select(func.count(Post.id)))
            
            if total > 100:
                await train_topic_model(min_posts=50)
                await update_refresh_record(refresh_id, topics_updated=True)
            else:
                st.warning("⚠️ Not enough posts for topic modeling yet (need 100+)")
        
        # Step 5: Calculate trends
        status_text.text("📈 Calculating trends...")
        progress_bar.progress(90)
        await calculate_trends_for_all_posts(recent_hours=6, previous_hours=12)
        
        await update_refresh_record(
            refresh_id,
            trends_calculated=True,
            status="completed"
        )
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Refresh complete!")
        
        return True
    
    except Exception as e:
        await update_refresh_record(
            refresh_id,
            status="failed",
            error_message=str(e)
        )
        st.error(f"❌ Error during refresh: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

# ============================================
# VISUALIZATION COMPONENTS
# ============================================

def render_sentiment_pie(stats):
    """Sentiment distribution pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[stats['positive'], stats['neutral'], stats['negative']],
        marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
        hole=0.4,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def render_subreddit_bar(subreddit_counts):
    """Subreddit distribution bar chart"""
    df = pd.DataFrame([
        {'Subreddit': k, 'Posts': v}
        for k, v in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    fig = px.bar(
        df,
        x='Posts',
        y='Subreddit',
        orientation='h',
        title="Posts by Subreddit",
        color='Posts',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    
    return fig

def render_timeline(timeline_data):
    """Timeline chart"""
    if not timeline_data:
        return None
    
    times = sorted(timeline_data.keys())
    positive = [timeline_data[t]['positive'] for t in times]
    neutral = [timeline_data[t]['neutral'] for t in times]
    negative = [timeline_data[t]['negative'] for t in times]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, y=positive,
        name='Positive',
        stackgroup='one',
        fillcolor='rgba(40, 167, 69, 0.5)',
        line=dict(color='#28a745')
    ))
    
    fig.add_trace(go.Scatter(
        x=times, y=neutral,
        name='Neutral',
        stackgroup='one',
        fillcolor='rgba(255, 193, 7, 0.5)',
        line=dict(color='#ffc107')
    ))
    
    fig.add_trace(go.Scatter(
        x=times, y=negative,
        name='Negative',
        stackgroup='one',
        fillcolor='rgba(220, 53, 69, 0.5)',
        line=dict(color='#dc3545')
    ))
    
    fig.update_layout(
        title="Posts Over Time (Last 24 Hours)",
        xaxis_title="Time",
        yaxis_title="Number of Posts",
        height=400,
        hovermode='x unified'
    )
    
    return fig

# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Real-Time News Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"**Monitoring:** {', '.join([f'r/{s}' for s in MONITORED_SUBREDDITS])}")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # REFRESH BUTTON
        if st.button("🔄 REFRESH DATA", type="primary", use_container_width=True):
            st.session_state['refreshing'] = True
            st.rerun()
        
        st.markdown("---")
        
        st.header("📊 Settings")
        time_window = st.selectbox(
            "Time Window",
            options=[6, 12, 24, 48],
            index=2,
            format_func=lambda x: f"Last {x} hours"
        )
        
        st.markdown("---")
        
        st.info(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")

        st.markdown("---")
        
        st.header("📜 Refresh History")
        
        if st.button("Show History", key="refresh_history_btn"):
            history = run_async(get_refresh_history(limit=5))
            st.session_state['refresh_history'] = history
        
        if st.session_state.get('refresh_history'):
            for h in st.session_state['refresh_history']:
                status_emoji = "✅" if h['status'] == 'completed' else "❌" if h['status'] == 'failed' else "⏳"
                with st.expander(f"{status_emoji} {h['started_at'].strftime('%m/%d %H:%M')}"):
                    st.write(f"**Status:** {h['status']}")
                    st.write(f"**Scraped:** {h['posts_scraped']}")
                    st.write(f"**Saved:** {h['posts_saved']}")
                    st.write(f"**Duration:** {h['duration_seconds']}s")

    # Handle refresh
    if st.session_state.get('refreshing', False):
        st.info("🔄 Starting refresh pipeline...")
        success = run_async(run_full_refresh_pipeline())
        
        if success:
            st.cache_data.clear()
            st.success("✅ Dashboard refreshed!")
        
        st.session_state['refreshing'] = False
        st.rerun()
    
    # Get data
    stats = get_dashboard_stats()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📰 Total Posts", f"{stats['total_posts']:,}")
    
    with col2:
        st.metric("🆕 Last 24h", stats['recent_posts'])
    
    with col3:
        sentiment_emoji = "😊" if stats['avg_sentiment'] > 0.1 else "😐" if stats['avg_sentiment'] > -0.1 else "😟"
        st.metric(f"{sentiment_emoji} Avg Sentiment", f"{stats['avg_sentiment']:.3f}")
    
    with col4:
        st.metric("🔍 Subreddits", len(stats['subreddit_counts']))
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_sentiment_pie(stats), use_container_width=True)
    
    with col2:
        st.plotly_chart(render_subreddit_bar(stats['subreddit_counts']), use_container_width=True)
    
    # Timeline
    st.markdown("---")
    timeline = get_posts_timeline(hours=time_window)
    timeline_fig = render_timeline(timeline)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Recent posts
    st.markdown("---")
    st.header("📋 Recent Posts")
    
    recent = get_recent_posts(hours=time_window, limit=20)
    
    for i, post in enumerate(recent[:10], 1):
        sentiment_class = post['sentiment'] or 'neutral'
        sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(sentiment_class, "😐")
        
        with st.expander(f"{i}. {sentiment_emoji} {post['title'][:80]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**r/{post['subreddit']}** • {post['created_at'].strftime('%Y-%m-%d %H:%M')}")
                if post['url']:
                    st.markdown(f"[🔗 View Post]({post['url']})")
            
            with col2:
                st.metric("Score", post['score'])
                st.metric("Comments", post['comments'])
    
    # AI Summary
    st.markdown("---")
    st.header("🤖 AI-Generated Summary")
    
    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            try:
                summary = run_async(generate_trending_summary(use_cache=False))
                st.success(summary)
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
