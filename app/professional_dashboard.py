"""
Professional News Analytics Platform
Unified dashboard: Reddit + Google News + AI Chat + Sentiment Analysis
Clean, professional design
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
import time
import uuid
from collections import Counter

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.db import DatabaseSession
from backend.models import Post, Topic
from config import MONITORED_SUBREDDITS
from scraper.reddit_scraper import workflow_1_refresh, workflow_2_query
from scraper.google_news_scraper import workflow_google_news, compare_sentiments
from backend.ingest import process_unprocessed_posts
from pipeline.embeddings import build_faiss_index
from pipeline.quick_pipeline import process_query_results
from backend.chat_storage import save_chat_interaction
from pipeline.summary import GeminiSummarizer

# Google News topics
GOOGLE_NEWS_TOPICS = [
    "artificial intelligence",
    "climate change",
    "space exploration",
    "cryptocurrency",
    "politics",
    "technology"
]

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="News Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PROFESSIONAL STYLING
# ============================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .chat-user {
        background: #f8f9fa;
        border-left: 3px solid #4285f4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .chat-assistant {
        background: #ffffff;
        border-left: 3px solid #34a853;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #4285f4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .success-box {
        background: #e6f4ea;
        border-left: 4px solid #34a853;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .warning-box {
        background: #fef7e0;
        border-left: 4px solid #fbbc04;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .stButton>button {
        font-weight: 500;
        border-radius: 6px;
        padding: 0.5rem 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

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
def get_all_stats():
    """Get comprehensive statistics"""
    async def fetch():
        async with DatabaseSession() as db:
            # Reddit
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
            
            # Google News
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
            
            # Sentiment by source
            def get_sentiment_dist(source):
                async def inner():
                    pos = await db.scalar(
                        select(func.count(Post.id)).where(
                            Post.source == source,
                            Post.sentiment_label == 'positive'
                        )
                    ) or 0
                    neg = await db.scalar(
                        select(func.count(Post.id)).where(
                            Post.source == source,
                            Post.sentiment_label == 'negative'
                        )
                    ) or 0
                    neu = await db.scalar(
                        select(func.count(Post.id)).where(
                            Post.source == source,
                            Post.sentiment_label == 'neutral'
                        )
                    ) or 0
                    return {'positive': pos, 'negative': neg, 'neutral': neu}
                return inner()
            
            reddit_sent_dist = await get_sentiment_dist('reddit')
            gnews_sent_dist = await get_sentiment_dist('google_news')
            
            return {
                'reddit': {
                    'total': reddit_total,
                    'recent_24h': reddit_24h,
                    'avg_sentiment': float(reddit_sentiment),
                    'sentiment_dist': reddit_sent_dist
                },
                'google_news': {
                    'total': gnews_total,
                    'recent_24h': gnews_24h,
                    'avg_sentiment': float(gnews_sentiment),
                    'sentiment_dist': gnews_sent_dist
                }
            }
    
    return run_async(fetch())

@st.cache_data(ttl=60)
def get_recent_posts(source, limit=10):
    """Get recent posts"""
    async def fetch():
        async with DatabaseSession() as db:
            result = await db.execute(
                select(Post).where(Post.source == source)
                .order_by(desc(Post.created_at))
                .limit(limit)
            )
            return [{
                'title': p.title,
                'source_name': p.subreddit or 'Unknown',
                'sentiment': p.sentiment_label or 'neutral',
                'score': p.score,
                'url': p.url,
                'created': p.created_at
            } for p in result.scalars().all()]
    
    return run_async(fetch())

# ============================================
# AI ASSISTANT
# ============================================

class NewsAssistant:
    """AI Assistant"""
    
    def __init__(self):
        self.gemini = GeminiSummarizer()
        self.gemini.initialize()
    
    async def answer(self, query: str):
        """Answer query"""
        # Search Reddit
        scrape_result = await workflow_2_query(query, search_all=True)
        
        if not scrape_result['success']:
            return {
                'answer': f"No recent posts found for '{query}'. Try rephrasing.",
                'confidence': 0.0,
                'posts_count': 0
            }
        
        # Get posts
        async with DatabaseSession() as db:
            scraped_ids = [p['post_id'] for p in scrape_result['posts'][:30]]
            result = await db.execute(
                select(Post).where(Post.post_id.in_(scraped_ids))
            )
            posts = result.scalars().all()
            post_ids = [p.id for p in posts]
        
        if not posts:
            await asyncio.sleep(2)
            async with DatabaseSession() as db:
                result = await db.execute(
                    select(Post).where(Post.post_id.in_(scraped_ids))
                )
                posts = result.scalars().all()
                post_ids = [p.id for p in posts]
        
        if not posts:
            return {
                'answer': f"Found posts but couldn't process them yet.",
                'confidence': 0.0,
                'posts_count': scrape_result['posts_found']
            }
        
        # Analyze
        analysis = await process_query_results(post_ids)
        
        # Generate answer
        context = "\n".join([
            f"{i}. [{p.sentiment_label}] r/{p.subreddit}: {p.title[:80]}"
            for i, p in enumerate(posts[:8], 1)
        ])
        
        prompt = f"""Answer based on Reddit posts.

Question: {query}

Posts:
{context}

Provide 3-4 sentence answer. Be factual and concise.

Answer:"""
        
        answer = self.gemini.generate(prompt, max_tokens=300)
        
        return {
            'answer': answer,
            'confidence': min(len(posts) / 20.0, 1.0),
            'posts_count': len(posts),
            'analysis': analysis
        }

# ============================================
# REFRESH PIPELINE
# ============================================

async def run_full_refresh():
    """Refresh all data"""
    progress = st.progress(0)
    status = st.empty()
    
    status.text("Scraping Reddit...")
    progress.progress(20)
    reddit_result = await workflow_1_refresh()
    
    status.text("Scraping Google News...")
    progress.progress(40)
    gnews_result = await workflow_google_news(GOOGLE_NEWS_TOPICS)
    
    status.text("Processing...")
    progress.progress(60)
    await process_unprocessed_posts(limit=None)
    
    status.text("Building index...")
    progress.progress(80)
    await build_faiss_index()
    
    progress.progress(100)
    status.text("Complete")
    
    return {'reddit': reddit_result, 'google_news': gnews_result}

# ============================================
# VISUALIZATIONS
# ============================================

def render_comparison_charts(stats):
    """Comparison charts"""
    
    # Sentiment comparison
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Reddit", "Google News")
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=stats['reddit']['avg_sentiment'],
            domain={'x': [0, 0.48], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#4285f4"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "#fee"},
                    {'range': [-0.3, 0.3], 'color': "#ffe"},
                    {'range': [0.3, 1], 'color': "#efe"}
                ]
            }
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=stats['google_news']['avg_sentiment'],
            domain={'x': [0.52, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#34a853"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "#fee"},
                    {'range': [-0.3, 0.3], 'color': "#ffe"},
                    {'range': [0.3, 1], 'color': "#efe"}
                ]
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=300, showlegend=False)
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-title">News Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time analysis of Reddit discussions and Google News coverage</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        if st.button("Refresh All Data", type="primary", use_container_width=True):
            with st.spinner("Refreshing..."):
                result = run_async(run_full_refresh())
                st.success(f"Updated: {result['reddit']['posts_saved']} Reddit + {result['google_news']['articles_saved']} News")
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("Data Sources")
        st.text("Reddit:")
        for sub in MONITORED_SUBREDDITS:
            st.text(f"  • r/{sub}")
        
        st.text("\nGoogle News Topics:")
        for topic in GOOGLE_NEWS_TOPICS[:4]:
            st.text(f"  • {topic}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "AI Assistant", "Comparison"])
    
    # ===== TAB 1: DASHBOARD =====
    with tab1:
        stats = get_all_stats()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Reddit Posts", f"{stats['reddit']['total']:,}", 
                     delta=f"+{stats['reddit']['recent_24h']} (24h)")
        
        with col2:
            st.metric("Google News", f"{stats['google_news']['total']:,}",
                     delta=f"+{stats['google_news']['recent_24h']} (24h)")
        
        with col3:
            st.metric("Reddit Sentiment", f"{stats['reddit']['avg_sentiment']:.3f}")
        
        with col4:
            st.metric("News Sentiment", f"{stats['google_news']['avg_sentiment']:.3f}")
        
        st.markdown('<div class="section-header">Sentiment Analysis</div>', unsafe_allow_html=True)
        
        st.plotly_chart(render_comparison_charts(stats), use_container_width=True)
        
        # Recent posts
        st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reddit")
            reddit_posts = get_recent_posts('reddit', 5)
            for post in reddit_posts:
                with st.expander(f"{post['title'][:60]}..."):
                    st.text(f"Source: r/{post['source_name']}")
                    st.text(f"Sentiment: {post['sentiment']}")
                    st.text(f"Score: {post['score']}")
                    if post['url']:
                        st.markdown(f"[View]({post['url']})")
        
        with col2:
            st.subheader("Google News")
            gnews_posts = get_recent_posts('google_news', 5)
            for post in gnews_posts:
                with st.expander(f"{post['title'][:60]}..."):
                    st.text(f"Source: {post['source_name']}")
                    st.text(f"Sentiment: {post['sentiment']}")
                    if post['url']:
                        st.markdown(f"[View]({post['url']})")
    
    # ===== TAB 2: AI ASSISTANT =====
    with tab2:
        st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Ask any question about current events. The assistant will search Reddit in real-time and provide an informed answer.
        </div>
        """, unsafe_allow_html=True)
        
        # Display conversation
        for msg in st.session_state['conversation']:
            if msg['type'] == 'user':
                st.markdown(f'<div class="chat-user"><strong>You:</strong> {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant"><strong>Assistant:</strong> {msg["text"]}</div>', unsafe_allow_html=True)
                if 'confidence' in msg:
                    st.caption(f"Confidence: {msg['confidence']:.1%} | Based on {msg.get('posts_count', 0)} posts")
        
        # Input
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query = st.text_input("Ask a question:", placeholder="e.g., What's the latest on artificial intelligence?")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            ask_btn = st.button("Ask", type="primary", use_container_width=True)
        
        if ask_btn and query:
            st.session_state['conversation'].append({'type': 'user', 'text': query})
            
            if 'bot' not in st.session_state:
                st.session_state['bot'] = NewsAssistant()
            
            with st.spinner("Searching and analyzing..."):
                result = run_async(st.session_state['bot'].answer(query))
            
            st.session_state['conversation'].append({
                'type': 'bot',
                'text': result['answer'],
                'confidence': result['confidence'],
                'posts_count': result['posts_count']
            })
            
            st.rerun()
    
    # ===== TAB 3: COMPARISON =====
    with tab3:
        st.markdown('<div class="section-header">Sentiment Comparison</div>', unsafe_allow_html=True)
        
        topic = st.text_input("Compare sentiment for topic:", placeholder="e.g., climate change")
        
        if st.button("Compare", type="primary"):
            with st.spinner("Analyzing..."):
                comparison = run_async(compare_sentiments(topic))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Reddit", f"{comparison['reddit']['avg']:.3f}")
                
                with col2:
                    st.metric("Google News", f"{comparison['google_news']['avg']:.3f}")
                
                with col3:
                    st.metric("Difference", f"{comparison['difference']:.3f}")
                
                if comparison['agreement']:
                    st.markdown('<div class="success-box"><strong>Sentiments Align:</strong> Reddit discussions match mainstream news coverage for this topic.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box"><strong>Sentiments Differ:</strong> There is a notable difference between Reddit sentiment and news coverage.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
