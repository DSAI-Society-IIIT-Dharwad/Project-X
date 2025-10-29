"""
Query Analyzer Dashboard - Workflow 2
Search Reddit by query and analyze results instantly
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import uuid
import time
from datetime import datetime
import sys
from pathlib import Path
from typing import List, Dict


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from backend.db import DatabaseSession
from backend.models import Post
from sqlalchemy import select
from scraper.reddit_scraper import workflow_2_query
from pipeline.quick_pipeline import process_query_results
from backend.chat_storage import save_query_analysis, get_query_history, update_session_activity

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="🔍 Query Analyzer",
    page_icon="🔍",
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
    .search-box {
        font-size: 1.2rem;
        padding: 1rem;
        border: 2px solid #667eea;
        border-radius: 10px;
        width: 100%;
    }
    .query-result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
    }
    .positive { background-color: #d4edda; color: #155724; }
    .negative { background-color: #f8d7da; color: #721c24; }
    .neutral { background-color: #fff3cd; color: #856404; }
    .keyword-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
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
# QUERY PROCESSING
# ============================================

async def process_query(query: str) -> Dict:
    """
    Complete query workflow:
    1. Search Reddit
    2. Process results
    3. Generate analysis
    4. SAVE TO DATABASE
    """
    start_time = time.time()
    
    results = {
        'query': query,
        'timestamp': datetime.now(),
        'success': False
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Search Reddit
        status_text.text(f"🔍 Searching Reddit for: '{query}'...")
        progress_bar.progress(20)
        
        scrape_result = await workflow_2_query(query)
        
        if not scrape_result['success']:
            st.error(f"❌ No posts found for query: {query}")
            return results
        
        st.success(f"✅ Found {scrape_result['posts_found']} posts ({scrape_result['posts_saved']} new)")
        results['posts_found'] = scrape_result['posts_found']
        
        # Step 2: Get post IDs
        status_text.text("📊 Retrieving posts...")
        progress_bar.progress(40)
        
        async with DatabaseSession() as db:
            result = await db.execute(
                select(Post.id).where(
                    Post.title.contains(query.split()[0])
                ).order_by(Post.created_at.desc()).limit(50)
            )
            post_ids = [row[0] for row in result.all()]
        
        if not post_ids:
            st.error("❌ No posts found in database")
            return results
        
        # Step 3: Quick processing
        status_text.text("⚡ Analyzing posts...")
        progress_bar.progress(60)
        
        analysis = await process_query_results(post_ids)
        
        # Step 4: SAVE TO DATABASE
        status_text.text("💾 Saving analysis...")
        progress_bar.progress(80)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        analysis_id = await save_query_analysis(
            query=query,
            analysis=analysis,
            post_ids=post_ids,
            processing_time_ms=processing_time_ms
        )
        
        # Update session activity
        session_id = st.session_state.get('session_id', str(uuid.uuid4()))
        st.session_state['session_id'] = session_id
        await update_session_activity(session_id, activity_type="query")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        results['success'] = True
        results['analysis'] = analysis
        results['analysis_id'] = analysis_id
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return results


# ============================================
# VISUALIZATION COMPONENTS
# ============================================

def render_sentiment_gauge(avg_sentiment: float):
    """Sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Sentiment"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "#f8d7da"},
                {'range': [-0.3, 0.3], 'color': "#fff3cd"},
                {'range': [0.3, 1], 'color': "#d4edda"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def render_sentiment_dist(sentiment_data: Dict):
    """Sentiment distribution bar chart"""
    df = pd.DataFrame([
        {'Sentiment': 'Positive', 'Count': sentiment_data['positive'], 'Color': '#28a745'},
        {'Sentiment': 'Neutral', 'Count': sentiment_data['neutral'], 'Color': '#ffc107'},
        {'Sentiment': 'Negative', 'Count': sentiment_data['negative'], 'Color': '#dc3545'}
    ])
    
    fig = px.bar(
        df,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'},
        title="Sentiment Distribution"
    )
    
    fig.update_layout(height=300, showlegend=False)
    return fig

def render_subreddit_pie(subreddit_data: Dict):
    """Subreddit distribution pie chart"""
    labels = list(subreddit_data.keys())
    values = list(subreddit_data.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3
    )])
    
    fig.update_layout(
        title="Posts by Subreddit",
        height=300
    )
    
    return fig

def render_engagement_bar(top_posts: List[Dict]):
    """Top posts by engagement"""
    df = pd.DataFrame([
        {
            'Title': p['title'][:30] + '...',
            'Score': p['score'],
            'Comments': p['comments']
        }
        for p in top_posts[:5]
    ])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Title'],
        y=df['Score'],
        name='Upvotes',
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Title'],
        y=df['Comments'],
        name='Comments',
        marker_color='#764ba2'
    ))
    
    fig.update_layout(
        title="Top 5 Posts by Engagement",
        barmode='group',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    """Main query analyzer application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Query Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Search Reddit and analyze posts instantly**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Recent Queries")
        
        if 'query_history' not in st.session_state:
            st.session_state['query_history'] = []
        
        if st.session_state['query_history']:
            for i, q in enumerate(reversed(st.session_state['query_history'][-5:]), 1):
                if st.button(f"{i}. {q}", key=f"hist_{i}"):
                    st.session_state['current_query'] = q
                    st.rerun()
        else:
            st.info("No queries yet")
            
        st.markdown("---")
        
        st.header("📜 Query History")
        
        if st.button("Load History", use_container_width=True):
            history = run_async(get_query_history(limit=10))
            st.session_state['query_history_full'] = history
        
        if st.session_state.get('query_history_full'):
            for h in st.session_state['query_history_full'][:5]:
                with st.expander(f"🔍 {h['query'][:30]}..."):
                    st.write(f"**Posts:** {h['posts_found']}")
                    st.write(f"**Sentiment:** {h['avg_sentiment']:.3f}")
                    st.write(f"**Keywords:** {', '.join(h['top_keywords'])}")
                    st.write(f"**Date:** {h['created_at'].strftime('%Y-%m-%d %H:%M')}")

        st.markdown("---")
        
        st.header("💡 Example Queries")
        examples = [
            "AI regulation",
            "climate change policy",
            "Ukraine conflict",
            "space exploration",
            "cryptocurrency news"
        ]
        
        for example in examples:
            if st.button(f"🔍 {example}", key=f"ex_{example}"):
                st.session_state['current_query'] = example
                st.rerun()
    
    # Main search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your search query:",
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., artificial intelligence, climate change, space news...",
            key="query_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        search_clicked = st.button("🔎 SEARCH", type="primary", use_container_width=True)
    
    # Process query
    if search_clicked and query:
        # Add to history
        if query not in st.session_state['query_history']:
            st.session_state['query_history'].append(query)
        
        st.markdown("---")
        
        # Run analysis
        results = run_async(process_query(query))
        
        if results['success']:
            analysis = results['analysis']
            
            # Store in session state
            st.session_state['last_analysis'] = analysis
            st.session_state['last_query'] = query
            
            st.markdown("---")
            
            # === RESULTS SECTION ===
            st.header(f"📊 Results for: '{query}'")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📰 Posts Found", analysis['total_posts'])
            
            with col2:
                sentiment_emoji = "😊" if analysis['sentiment']['average_score'] > 0.2 else "😐" if analysis['sentiment']['average_score'] > -0.2 else "😟"
                st.metric(f"{sentiment_emoji} Avg Sentiment", f"{analysis['sentiment']['average_score']:.3f}")
            
            with col3:
                st.metric("👍 Total Upvotes", f"{analysis['engagement']['total_upvotes']:,}")
            
            with col4:
                st.metric("💬 Total Comments", f"{analysis['engagement']['total_comments']:,}")
            
            st.markdown("---")
            
            # Charts row 1
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(render_sentiment_gauge(analysis['sentiment']['average_score']), use_container_width=True)
            
            with col2:
                st.plotly_chart(render_sentiment_dist(analysis['sentiment']), use_container_width=True)
            
            # Charts row 2
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(render_subreddit_pie(analysis['subreddits']), use_container_width=True)
            
            with col2:
                st.plotly_chart(render_engagement_bar(analysis['top_posts']), use_container_width=True)
            
            # Keywords section
            st.markdown("---")
            st.subheader("🔑 Top Keywords")
            
            keyword_html = " ".join([
                f'<span class="keyword-tag">{kw}</span>'
                for kw in analysis['keywords'][:15]
            ])
            st.markdown(keyword_html, unsafe_allow_html=True)
            
            # AI Summary
            st.markdown("---")
            st.subheader("🤖 AI-Generated Summary")
            
            if 'summary' in analysis:
                st.success(analysis['summary'])
            else:
                st.info("Summary not generated. Enable in pipeline settings.")
            
            # Top posts
            st.markdown("---")
            st.subheader("📋 Top Posts")
            
            for i, post in enumerate(analysis['top_posts'], 1):
                sentiment_class = post['sentiment'] or 'neutral'
                sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(sentiment_class, "😐")
                
                with st.expander(f"{i}. {sentiment_emoji} {post['title'][:80]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**r/{post['subreddit']}**")
                        if post['url']:
                            st.markdown(f"[🔗 View on Reddit]({post['url']})")
                    
                    with col2:
                        st.metric("Score", post['score'])
                        st.metric("Comments", post['comments'])
                        st.markdown(f"<span class='sentiment-badge {sentiment_class}'>{sentiment_class.upper()}</span>", unsafe_allow_html=True)
    
    elif not query and not st.session_state.get('last_analysis'):
        # Welcome screen
        st.info("👆 Enter a search query above to analyze Reddit posts")
        
        st.markdown("### 🎯 How it works:")
        st.markdown("""
        1. **Enter a query** - Type any topic you want to analyze
        2. **We search Reddit** - Find relevant posts from monitored subreddits
        3. **Instant analysis** - Get sentiment, trends, and insights
        4. **AI summary** - Receive an AI-generated overview
        """)
        
        st.markdown("### 💡 Tips:")
        st.markdown("""
        - Use specific keywords (e.g., "AI regulation" vs "technology")
        - Combine topics (e.g., "climate change policy 2025")
        - Try different phrasings if you don't get results
        """)
    
    # Show previous results if available
    elif st.session_state.get('last_analysis') and not search_clicked:
        st.info(f"Showing previous results for: '{st.session_state.get('last_query')}'")
        st.markdown("*Enter a new query to search again*")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
