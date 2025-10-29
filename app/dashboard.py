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
from config import MONITORED_SUBREDDITS, ENABLE_GOOGLE_NEWS
from scraper.reddit_scraper import workflow_1_refresh
from scraper.google_news_scraper import workflow_1_refresh as google_news_refresh

# Optional ML imports - handle gracefully if not available
try:
    from backend.ingest import process_unprocessed_posts
    ML_AVAILABLE = True
except ImportError:
    process_unprocessed_posts = None
    ML_AVAILABLE = False

try:
    from pipeline.embeddings import build_faiss_index
    from pipeline.topic_model import train_topic_model, assign_topics_to_new_posts
    from pipeline.trend_score import calculate_trends_for_all_posts
    from pipeline.summary import generate_trending_summary
except ImportError:
    def build_faiss_index():
        return None
    def train_topic_model(*args, **kwargs):
        return None
    def assign_topics_to_new_posts():
        return None
    def calculate_trends_for_all_posts(*args, **kwargs):
        return None
    def generate_trending_summary(*args, **kwargs):
        return "ML features not available - install required dependencies"

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
            
            # Source distribution
            source_result = await db.execute(
                select(Post.source, func.count(Post.id))
                .group_by(Post.source)
            )
            source_counts = dict(source_result.all())
            
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
                'subreddit_counts': subreddit_counts,
                'source_counts': source_counts
            }
    
    return run_async(fetch())

@st.cache_data(ttl=60)
def get_recent_posts(hours=24, limit=50, source=None):
    """Get recent posts"""
    async def fetch():
        async with DatabaseSession() as db:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            query = select(Post).where(Post.created_at >= cutoff)
            
            if source:
                query = query.where(Post.source == source)
            
            result = await db.execute(
                query.order_by(desc(Post.created_at)).limit(limit)
            )
            posts = result.scalars().all()
            
            return [
                {
                    'title': p.title,
                    'source': p.source,
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
        # Step 1: Scrape Reddit
        status_text.text("📡 Scraping Reddit...")
        progress_bar.progress(10)
        reddit_result = await workflow_1_refresh()
        
        total_scraped = reddit_result.get('posts_scraped', 0)
        total_saved = reddit_result.get('posts_saved', 0)
        total_skipped = reddit_result.get('posts_skipped', 0)
        
        if reddit_result.get('success'):
            st.success(f"✅ Reddit: {total_scraped} posts ({total_saved} new)")
        
        # Step 1b: Scrape Google News if enabled
        if ENABLE_GOOGLE_NEWS:
            status_text.text("📰 Scraping Google News...")
            progress_bar.progress(15)
            google_result = await google_news_refresh()
            
            if google_result.get('success'):
                total_scraped += google_result.get('posts_scraped', 0)
                total_saved += google_result.get('posts_saved', 0)
                total_skipped += google_result.get('posts_skipped', 0)
                st.success(f"✅ Google News: {google_result.get('posts_scraped', 0)} articles ({google_result.get('posts_saved', 0)} new)")
        
        await update_refresh_record(
            refresh_id,
            posts_scraped=total_scraped,
            posts_saved=total_saved,
            posts_skipped=total_skipped
        )
        
        # Step 2: Process posts (if ML available)
        if ML_AVAILABLE and process_unprocessed_posts:
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
        else:
            progress_bar.progress(90)
            st.info("ℹ️ ML processing skipped - basic functionality only")
        
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
    if not subreddit_counts:
        # Return empty chart if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        fig.update_layout(title="Posts by Source", height=400)
        return fig
    
    df = pd.DataFrame([
        {'Source': k, 'Posts': v}
        for k, v in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        fig.update_layout(title="Posts by Source", height=400)
        return fig
    
    fig = px.bar(
        df,
        x='Posts',
        y='Source',
        orientation='h',
        title="Posts by Source",
        color='Posts',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    
    return fig

def render_source_distribution(source_counts):
    """Source distribution pie chart"""
    if not source_counts:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(source_counts.keys()),
        values=list(source_counts.values()),
        hole=0.4,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title="News Source Distribution",
        height=400,
        showlegend=True
    )
    
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
    sources_text = f"**Reddit:** {', '.join([f'r/{s}' for s in MONITORED_SUBREDDITS])}"
    if ENABLE_GOOGLE_NEWS:
        sources_text += " | **Google News:** Enabled"
    st.markdown(sources_text)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # REFRESH BUTTON
        if st.button("🔄 REFRESH DATA", type="primary", use_container_width=True):
            st.session_state['refreshing'] = True
            st.rerun()
        
        st.markdown("---")
        
        st.header("🔍 Custom Search")
        with st.form("search_form"):
            search_query = st.text_input("Search Topic")
            search_submitted = st.form_submit_button("Search News")
            
            if search_submitted and search_query:
                st.session_state['search_query'] = search_query
                st.session_state['searching'] = True
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

    # Handle custom search
    if st.session_state.get('searching', False):
        search_query = st.session_state.get('search_query', '')
        if search_query:
            st.info(f"🔍 Searching for: {search_query}")
            try:
                from scraper.google_news_scraper import scrape_google_news
                articles = scrape_google_news(query=search_query, limit=20)
                
                if articles:
                    st.success(f"✅ Found {len(articles)} articles")
                    st.subheader("📰 Search Results")
                    
                    for i, article in enumerate(articles[:10], 1):
                        with st.expander(f"{i}. {article['title'][:70]}..."):
                            st.markdown(f"**Source:** {article['subreddit']}")
                            st.markdown(f"**Published:** {article['created_at'].strftime('%Y-%m-%d %H:%M')}")
                            st.markdown(f"[🔗 Read Article]({article['url']})")
                    
                    if ML_AVAILABLE:
                        st.subheader("📊 Sentiment Analysis of Search Results")
                        st.info("Sentiment analysis would be shown here")
                else:
                    st.warning("No articles found for this query")
            except Exception as e:
                st.error(f"Error searching: {e}")
        
        st.session_state['searching'] = False
    
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
        st.metric(f"{sentiment_emoji} Avg Sentiment", f"{stats['avg_sentiment']:.2f}")
    
    with col4:
        source_label = "🔍 Sources" if ENABLE_GOOGLE_NEWS else "🔍 Subreddits"
        source_count = len(stats.get('source_counts', stats['subreddit_counts']))
        st.metric(source_label, source_count)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(render_sentiment_pie(stats), use_container_width=True)
    
    with col2:
        if ENABLE_GOOGLE_NEWS and stats.get('source_counts'):
            st.plotly_chart(render_source_distribution(stats['source_counts']), use_container_width=True)
        else:
            st.plotly_chart(render_subreddit_bar(stats['subreddit_counts']), use_container_width=True)
    
    # Timeline
    st.markdown("---")
    timeline = get_posts_timeline(hours=time_window)
    timeline_fig = render_timeline(timeline)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Recent posts - Show Reddit and Google News side by side
    st.markdown("---")
    st.header("📋 Recent Posts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📰 Reddit News")
        reddit_posts = get_recent_posts(hours=time_window, limit=10, source='reddit')
        
        if reddit_posts:
            for i, post in enumerate(reddit_posts, 1):
                sentiment_class = post['sentiment'] or 'neutral'
                sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(sentiment_class, "😐")
                
                with st.expander(f"{i}. {sentiment_emoji} {post['title'][:60]}..."):
                    st.markdown(f"**r/{post['subreddit']}** • {post['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    if post['url']:
                        st.markdown(f"[🔗 View Post]({post['url']})")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Score", post['score'])
                    with col_b:
                        st.metric("Comments", post['comments'])
        else:
            st.info("No Reddit posts in this time window")
    
    with col2:
        st.subheader("🌐 Google News")
        google_posts = get_recent_posts(hours=time_window, limit=10, source='google_news')
        st.markdown(f"📊 Found {len(google_posts)} articles")
        
        if google_posts:
            for i, post in enumerate(google_posts, 1):
                sentiment_class = post['sentiment'] or 'neutral'
                sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(sentiment_class, "😐")
                
                with st.expander(f"{i}. {sentiment_emoji} {post['title'][:60]}..."):
                    st.markdown(f"**{post['subreddit']}** • {post['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    if post['url']:
                        st.markdown(f"[🔗 View Article]({post['url']})")
                    
                    st.markdown(f"**Source:** {post['subreddit']}")
        else:
            st.info("No Google News articles in this time window")
    
    # AI Summary
    st.markdown("---")
    st.header("🤖 AI-Generated Summary")
    
    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            try:
                if ML_AVAILABLE:
                    summary = run_async(generate_trending_summary(use_cache=False))
                    st.success(summary)
                else:
                    st.warning("Summary generation requires ML features. Please ensure all dependencies are installed.")
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                st.info("💡 Try refreshing the data first to ensure there are posts with sentiment analysis")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
