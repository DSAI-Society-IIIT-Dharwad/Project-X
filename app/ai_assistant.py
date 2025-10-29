"""
Intelligent AI News Assistant
Chat Mode + Sentiment Analysis Mode
Automatically searches Reddit for any query
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
from datetime import datetime
import sys
from pathlib import Path
import time
import uuid
from collections import Counter
from sqlalchemy import or_, func
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.db import DatabaseSession
from backend.models import Post
from sqlalchemy import select
from scraper.reddit_scraper import workflow_2_query
from pipeline.quick_pipeline import process_query_results
from backend.chat_storage import save_chat_interaction, get_chat_history
from pipeline.summary import GeminiSummarizer

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="🤖 AI News Assistant",
    page_icon="🤖",
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
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
    }
    .bot-message {
        background: #f8f9fa;
        color: #333;
        margin-right: 15%;
        border-left: 4px solid #667eea;
    }
    .processing-step {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
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

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'chat'

# ============================================
# HELPER FUNCTIONS
# ============================================

def run_async(coro):
    """Run async function in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ============================================
# INTELLIGENT CHATBOT
# ============================================

class IntelligentNewsBot:
    """Smart chatbot that automatically searches Reddit for any query"""
    
    def __init__(self):
        self.gemini = GeminiSummarizer()
        self.gemini.initialize()
    
    async def answer_question(self, user_query: str):
        """
        Main intelligence:
        1. Search Reddit for the query
        2. Scrape posts
        3. Analyze them
        4. Generate answer
        """
        start_time = time.time()
        
        # Container for steps
        steps_container = st.empty()
        
        # Step 1: Search Reddit
        with steps_container.container():
            st.markdown('<div class="processing-step">🔍 Step 1/4: Searching ALL of Reddit...</div>', unsafe_allow_html=True)
        
        scrape_result = await workflow_2_query(user_query, search_all=True)
        
        if not scrape_result['success'] or scrape_result['posts_found'] == 0:
            return {
                'answer': f"I searched Reddit but couldn't find recent posts about '{user_query}'. This topic might not be actively discussed right now, or try rephrasing your question.",
                'posts_found': 0,
                'analysis': None,
                'confidence': 0.0
            }
        
        # Step 2: Get scraped posts
        with steps_container.container():
            st.markdown('<div class="processing-step">📊 Step 2/4: Loading posts...</div>', unsafe_allow_html=True)
        
        async with DatabaseSession() as db:
            # Get posts that match the query
            scraped_post_ids = [p['post_id'] for p in scrape_result['posts'][:50]]
            

            result = await db.execute(
                select(Post).where(
                    Post.post_id.in_(scraped_post_ids)
                )
            )
            posts = result.scalars().all()
            post_ids = [p.id for p in posts]
        
        if not posts:
            await asyncio.sleep(2)

            async with DatabaseSession() as db:
                result = await db.execute(
                    select(Post).where(
                        Post.post_id.in_(scraped_post_ids)
                    )
                )
                posts = result.scalars().all()
                post_ids = [p.id for p in posts]
                
        if not posts:
            return {
                'answer': f"Found {scrape_result['posts_found']} posts about '{user_query}' but couldn't process them. Try again in a moment.",
                'posts_found': scrape_result['posts_found'],
                'analysis': None,
                'confidence': 0.0
            }
        # Step 3: Quick analysis
        with steps_container.container():
            st.markdown('<div class="processing-step">⚡ Step 3/4: Analyzing posts...</div>', unsafe_allow_html=True)
        
        analysis = await process_query_results(post_ids)
        
        # Step 4: Generate answer
        with steps_container.container():
            st.markdown('<div class="processing-step">🤖 Step 4/4: Generating answer...</div>', unsafe_allow_html=True)
        
        # Build context from analysis
        context = self._build_context(posts, analysis)
        
        # Generate answer
        prompt = f"""You are a helpful news assistant that answers questions based on recent Reddit discussions.

User Question: {user_query}

Recent Reddit Posts Context:
{context}

Analysis Summary:
- Total posts found: {analysis['total_posts']}
- Average sentiment: {analysis['sentiment']['average_score']:.2f}
- Top keywords: {', '.join(analysis['keywords'][:8])}
- Main subreddits: {', '.join(list(analysis['subreddits'].keys())[:5])}

Instructions:
1. Answer the user's question directly based on the Reddit posts
2. Be conversational and friendly (3-5 sentences)
3. Mention specific trends or sentiments if relevant
4. If sentiment is notable, mention it
5. Keep it concise but informative

Answer:"""
        
        answer = self.gemini.generate(prompt, max_tokens=400)
        
        # Clear processing steps
        steps_container.empty()
        
        # Calculate metrics
        response_time_ms = int((time.time() - start_time) * 1000)
        confidence = min(analysis['total_posts'] / 20.0, 1.0)
        
        # Save to database
        await save_chat_interaction(
            user_query=user_query,
            bot_response=answer,
            relevant_post_ids=post_ids[:10],
            confidence_score=confidence,
            session_id=st.session_state['session_id'],
            response_time_ms=response_time_ms,
            query_type="intelligent_search"
        )
        
        return {
            'answer': answer,
            'posts_found': analysis['total_posts'],
            'analysis': analysis,
            'confidence': confidence,
            'posts': posts[:5]
        }
    
    def _build_context(self, posts, analysis):
        """Build context from posts and analysis"""
        context_parts = []
        
        for i, post in enumerate(posts[:8], 1):
            sentiment = ""
            if post.sentiment_label:
                sentiment = f"[{post.sentiment_label}]"
            
            context_parts.append(
                f"{i}. {sentiment} r/{post.subreddit}: {post.title}"
            )
        
        return "\n".join(context_parts)

# ============================================
# SENTIMENT ANALYSIS MODE
# ============================================

async def analyze_sentiment_only(query: str):
    """Pure sentiment analysis mode"""
    start_time = time.time()
    
    progress = st.progress(0)
    status = st.empty()
    
    # Step 1: Search Reddit
    status.text(f"🔍 Searching ALL of Reddit for: '{query}'...")
    progress.progress(20)
    
    scrape_result = await workflow_2_query(query, search_all=True)
    
    if not scrape_result['success'] or scrape_result['posts_found'] == 0:
        return None
    
    # Step 2: Get posts
    status.text("📊 Analyzing sentiment...")
    progress.progress(50)
    
    async with DatabaseSession() as db:
        search_terms = query.lower().split()
        conditions = []
        for term in search_terms:
            conditions.append(or_(
                func.lower(Post.title).contains(term),
                func.lower(Post.content).contains(term)
            ))
        result = await db.execute(
            select(Post).where(
                or_(*conditions)
            ).order_by(Post.created_at.desc()).limit(100)
        )
        posts = result.scalars().all()
        post_ids = [p.id for p in posts]
    
    # Step 3: Analyze
    analysis = await process_query_results(post_ids)
    
    progress.progress(80)
    
    # Calculate additional sentiment metrics
    sentiment_by_subreddit = {}
    sentiment_over_time = {}
    
    for post in posts:
        # By subreddit
        if post.subreddit not in sentiment_by_subreddit:
            sentiment_by_subreddit[post.subreddit] = {'pos': 0, 'neg': 0, 'neu': 0, 'total': 0}
        
        sentiment_by_subreddit[post.subreddit]['total'] += 1
        if post.sentiment_label == 'positive':
            sentiment_by_subreddit[post.subreddit]['pos'] += 1
        elif post.sentiment_label == 'negative':
            sentiment_by_subreddit[post.subreddit]['neg'] += 1
        else:
            sentiment_by_subreddit[post.subreddit]['neu'] += 1
        
        # Over time
        date = post.created_at.date()
        if date not in sentiment_over_time:
            sentiment_over_time[date] = {'pos': 0, 'neg': 0, 'neu': 0}
        
        if post.sentiment_label == 'positive':
            sentiment_over_time[date]['pos'] += 1
        elif post.sentiment_label == 'negative':
            sentiment_over_time[date]['neg'] += 1
        else:
            sentiment_over_time[date]['neu'] += 1
    
    progress.progress(100)
    status.text("✅ Complete!")
    
    return {
        'analysis': analysis,
        'posts': posts,
        'sentiment_by_subreddit': sentiment_by_subreddit,
        'sentiment_over_time': sentiment_over_time
    }

# ============================================
# VISUALIZATION
# ============================================

def render_analysis_mini(analysis, message_id):
    """Compact analysis view for chat"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📰 Posts", analysis['total_posts'])
    
    with col2:
        sentiment_emoji = "😊" if analysis['sentiment']['average_score'] > 0.2 else "😐" if analysis['sentiment']['average_score'] > -0.2 else "😟"
        st.metric(f"{sentiment_emoji} Sentiment", f"{analysis['sentiment']['average_score']:.2f}")
    
    with col3:
        st.metric("💬 Engagement", f"{analysis['engagement']['total_comments']:,}")
    
    with st.expander("📊 View Detailed Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[
                    analysis['sentiment']['positive'],
                    analysis['sentiment']['neutral'],
                    analysis['sentiment']['negative']
                ],
                marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
                hole=0.4
            )])
            fig.update_layout(title="Sentiment", height=250, showlegend=True)
            st.plotly_chart(fig, use_container_width=True, key=f"sentiment_chart_{message_id}")
        
        with col2:
            df = pd.DataFrame([
                {'Subreddit': k, 'Posts': v}
                for k, v in list(analysis['subreddits'].items())[:5]
            ])
            fig = px.bar(df, x='Posts', y='Subreddit', orientation='h', title="Top Subreddits")
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key=f"subreddit_chart_{message_id}")
        
        st.markdown("**🔑 Keywords:**")
        keywords_html = " ".join([
            f'<span style="background:#667eea;color:white;padding:0.3rem 0.8rem;border-radius:15px;margin:0.2rem;display:inline-block;font-size:0.85rem;">{kw}</span>'
            for kw in analysis['keywords'][:12]
        ])
        st.markdown(keywords_html, unsafe_allow_html=True)


def render_sentiment_dashboard(data, query):
    """Full sentiment analysis dashboard"""
    
    analysis = data['analysis']
    
    st.header(f"📊 Sentiment Analysis: {query}")
    
    # Big metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📰 Total Posts", analysis['total_posts'])
    
    with col2:
        pos_pct = (analysis['sentiment']['positive'] / analysis['total_posts']) * 100
        st.metric("😊 Positive", f"{pos_pct:.1f}%", 
                  delta=f"{analysis['sentiment']['positive']} posts")
    
    with col3:
        neg_pct = (analysis['sentiment']['negative'] / analysis['total_posts']) * 100
        st.metric("😟 Negative", f"{neg_pct:.1f}%",
                  delta=f"{analysis['sentiment']['negative']} posts",
                  delta_color="inverse")
    
    with col4:
        neu_pct = (analysis['sentiment']['neutral'] / analysis['total_posts']) * 100
        st.metric("😐 Neutral", f"{neu_pct:.1f}%",
                  delta=f"{analysis['sentiment']['neutral']} posts")
    
    st.markdown("---")
    
    # Sentiment gauge
    avg_sentiment = analysis['sentiment']['average_score']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Sentiment Score"},
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
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="sentiment_gauge_main")
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment over time
        if data['sentiment_over_time']:
            dates = sorted(data['sentiment_over_time'].keys())
            pos = [data['sentiment_over_time'][d]['pos'] for d in dates]
            neg = [data['sentiment_over_time'][d]['neg'] for d in dates]
            neu = [data['sentiment_over_time'][d]['neu'] for d in dates]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=pos, name='Positive', 
                                     line=dict(color='#28a745'), fill='tozeroy'))
            fig.add_trace(go.Scatter(x=dates, y=neu, name='Neutral',
                                     line=dict(color='#ffc107'), fill='tozeroy'))
            fig.add_trace(go.Scatter(x=dates, y=neg, name='Negative',
                                     line=dict(color='#dc3545'), fill='tozeroy'))
            
            fig.update_layout(title="Sentiment Over Time", height=350)
            st.plotly_chart(fig, use_container_width=True, key="sentiment_timeline_main")
    
    with col2:
        # Sentiment by subreddit
        if data['sentiment_by_subreddit']:
            sub_data = []
            for sub, counts in data['sentiment_by_subreddit'].items():
                if counts['total'] > 2:
                    pos_pct = (counts['pos'] / counts['total']) * 100
                    sub_data.append({
                        'Subreddit': f"r/{sub}",
                        'Positive %': pos_pct,
                        'Total': counts['total']
                    })
            
            df = pd.DataFrame(sub_data).sort_values('Positive %', ascending=False).head(10)
            
            fig = px.bar(df, x='Positive %', y='Subreddit', orientation='h',
                        title="Most Positive Subreddits",
                        color='Positive %',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True, key="sentiment_by_sub_main")
    
    # Keywords by sentiment
    st.markdown("---")
    st.subheader("🔑 Keywords by Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**😊 Positive Keywords:**")
        pos_posts = [p for p in data['posts'] if p.sentiment_label == 'positive']
        if pos_posts:
            words = []
            for p in pos_posts[:20]:
                words.extend(p.title.lower().split())
            common = Counter(words).most_common(8)
            for word, count in common:
                if len(word) > 4:
                    st.markdown(f"- `{word}` ({count})")
    
    with col2:
        st.markdown("**😐 Neutral Keywords:**")
        neu_posts = [p for p in data['posts'] if p.sentiment_label == 'neutral']
        if neu_posts:
            words = []
            for p in neu_posts[:20]:
                words.extend(p.title.lower().split())
            common = Counter(words).most_common(8)
            for word, count in common:
                if len(word) > 4:
                    st.markdown(f"- `{word}` ({count})")
    
    with col3:
        st.markdown("**😟 Negative Keywords:**")
        neg_posts = [p for p in data['posts'] if p.sentiment_label == 'negative']
        if neg_posts:
            words = []
            for p in neg_posts[:20]:
                words.extend(p.title.lower().split())
            common = Counter(words).most_common(8)
            for word, count in common:
                if len(word) > 4:
                    st.markdown(f"- `{word}` ({count})")
    
    # Top posts
    st.markdown("---")
    st.subheader("📋 Sample Posts")
    
    for i, post in enumerate(data['posts'][:10], 1):
        sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(post.sentiment_label, "😐")
        with st.expander(f"{i}. {sentiment_emoji} {post.title[:60]}..."):
            st.write(f"**r/{post.subreddit}** • {post.score} upvotes • {post.num_comments} comments")
            if post.url:
                st.markdown(f"[🔗 View on Reddit]({post.url})")

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI News Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Mode")
        
        mode = st.radio(
            "Choose Mode:",
            options=['chat', 'sentiment'],
            format_func=lambda x: "💬 Chat Mode" if x == 'chat' else "📊 Sentiment Analysis",
            key='mode_selector'
        )
        
        st.session_state['mode'] = mode
        
        st.markdown("---")
        
        if mode == 'chat':
            st.info("""
            **💬 Chat Mode:**
            Ask any question and get conversational answers based on Reddit discussions.
            
            Searches ALL of Reddit in real-time!
            """)
        else:
            st.info("""
            **📊 Sentiment Analysis:**
            Enter a topic to see detailed sentiment breakdown, trends, and public opinion.
            
            Analyzes sentiment across all Reddit discussions!
            """)
        
        st.markdown("---")
        
        st.header("💭 Example Queries")
        
        if mode == 'chat':
            examples = [
                "What's happening with SpaceX?",
                "Latest AI regulation news",
                "Climate change discussions",
                "Ukraine war updates",
                "Cryptocurrency trends"
            ]
        else:
            examples = [
                "climate change",
                "electric vehicles",
                "artificial intelligence",
                "remote work",
                "cryptocurrency"
            ]
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"ex_{example}", use_container_width=True):
                st.session_state['example_query'] = example
                st.rerun()
        
        st.markdown("---")
        
        if mode == 'chat':
            st.header("📜 Chat History")
            
            if st.button("Load History", use_container_width=True):
                history = run_async(get_chat_history(st.session_state['session_id'], limit=10))
                st.session_state['chat_history'] = history
            
            if st.session_state.get('chat_history'):
                for h in st.session_state['chat_history'][:5]:
                    with st.expander(f"💬 {h['user_query'][:25]}..."):
                        st.write(f"**Q:** {h['user_query']}")
                        st.write(f"**Confidence:** {h['confidence']:.1%}")
                        st.write(f"**Time:** {h['created_at'].strftime('%H:%M:%S')}")
            
            st.markdown("---")
            
            if st.button("🔄 Clear Conversation", use_container_width=True):
                st.session_state['conversation'] = []
                st.rerun()
    
    # Main content area
    if st.session_state['mode'] == 'chat':
        # CHAT MODE
        st.markdown("**Ask me anything! I'll search ALL of Reddit and provide answers.**")
        st.markdown("---")
        
        st.header("💬 Conversation")
        
        # Display conversation
        for idx, msg in enumerate(st.session_state['conversation']):
            if msg['type'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Assistant:</strong><br>{msg["text"]}</div>', unsafe_allow_html=True)
                
                if 'analysis' in msg and msg['analysis']:
                    render_analysis_mini(msg['analysis'], message_id=idx)
                
                if 'confidence' in msg:
                    st.info(f"🎯 Confidence: {msg['confidence']:.1%} | 📊 Based on {msg.get('posts_found', 0)} Reddit posts")
        
        # Input
        st.markdown("---")
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me anything:",
                value=st.session_state.get('example_query', ''),
                placeholder="e.g., What's the latest news about artificial intelligence?",
                key="user_input_chat"
            )
            
            if 'example_query' in st.session_state:
                del st.session_state['example_query']
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.button("🚀 Ask", type="primary", use_container_width=True)
        
        if submit and user_input:
            st.session_state['conversation'].append({
                'type': 'user',
                'text': user_input
            })
            
            if 'bot' not in st.session_state:
                st.session_state['bot'] = IntelligentNewsBot()
            
            with st.spinner("🤔 Searching Reddit and analyzing..."):
                result = run_async(st.session_state['bot'].answer_question(user_input))
            
            st.session_state['conversation'].append({
                'type': 'bot',
                'text': result['answer'],
                'analysis': result['analysis'],
                'confidence': result['confidence'],
                'posts_found': result['posts_found']
            })
            
            st.rerun()
    
    else:
        # SENTIMENT ANALYSIS MODE
        st.markdown("**Enter any topic to analyze sentiment across ALL of Reddit**")
        st.markdown("---")
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            topic = st.text_input(
                "Topic to analyze:",
                value=st.session_state.get('example_query', ''),
                placeholder="e.g., climate change, artificial intelligence, cryptocurrency",
                key="topic_input"
            )
            
            if 'example_query' in st.session_state:
                del st.session_state['example_query']
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("📊 Analyze", type="primary", use_container_width=True)
        
        if analyze_btn and topic:
            st.markdown("---")
            
            data = run_async(analyze_sentiment_only(topic))
            
            if data:
                render_sentiment_dashboard(data, topic)
            else:
                st.error(f"No posts found for: {topic}")

if __name__ == "__main__":
    main()
