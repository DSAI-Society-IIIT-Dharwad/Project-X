"""
Quick Pipeline for Query-Based Analysis
Fast processing for Workflow 2 (doesn't build full topic models)
"""

import numpy as np
from typing import List, Dict, Optional
import asyncio
from sqlalchemy import select, desc
from collections import Counter

from backend.db import DatabaseSession
from backend.models import Post
from pipeline.preprocess import preprocess_text
from pipeline.embeddings import generate_embeddings_batch, embedding_to_list
from pipeline.sentiment import analyze_sentiment_batch

# ============================================
# QUICK PROCESSING
# ============================================

async def quick_process_posts(
    post_ids: List[int],
    generate_summary: bool = True
) -> Dict:
    """
    Fast processing for specific posts (Workflow 2)
    
    Args:
        post_ids: List of post IDs to process
        generate_summary: Whether to generate AI summary
    
    Returns:
        Analysis results dictionary
    """
    print(f"\n⚡ Quick Pipeline: Processing {len(post_ids)} posts...")
    
    async with DatabaseSession() as db:
        # Get posts
        result = await db.execute(
            select(Post).where(Post.id.in_(post_ids))
        )
        posts = result.scalars().all()
        
        if not posts:
            return {'error': 'No posts found'}
        
        print(f"📊 Found {len(posts)} posts")
        
        # Step 1: Preprocess
        print("1️⃣ Preprocessing...")
        texts = []
        for post in posts:
            full_text = f"{post.title} {post.content or ''}"
            cleaned = preprocess_text(full_text)
            texts.append(cleaned)
        
        # Step 2: Generate embeddings (for similarity, not storage)
        print("2️⃣ Generating embeddings...")
        embeddings = generate_embeddings_batch(texts, preprocess=False, batch_size=32)
        
        # Store embeddings in database
        for post, embedding in zip(posts, embeddings):
            post.embedding = embedding_to_list(embedding)
        
        await db.commit()
        
        # Step 3: Sentiment analysis
        print("3️⃣ Analyzing sentiment...")
        sentiments = analyze_sentiment_batch(texts, preprocess_texts=False, batch_size=16)
        
        # Store sentiment
        for post, sentiment in zip(posts, sentiments):
            post.sentiment_label = sentiment['label']
            post.sentiment_score = sentiment['sentiment_score']
            post.is_processed = True
        
        await db.commit()
        
        # Step 4: Quick analysis (no topic modeling)
        print("4️⃣ Generating quick insights...")
        analysis = await analyze_quick_insights(posts, sentiments, texts)
        
        # Step 5: Generate summary (optional)
        if generate_summary:
            print("5️⃣ Generating AI summary...")
            from pipeline.summary import summarizer
            summary = generate_query_summary(posts, analysis)
            analysis['summary'] = summary
        
        print("✅ Quick processing complete!")
        
        return analysis


async def analyze_quick_insights(
    posts: List[Post],
    sentiments: List[Dict],
    texts: List[str]
) -> Dict:
    """
    Generate quick insights without full topic modeling
    
    Returns:
        Dictionary with analysis results
    """
    # Sentiment distribution
    sentiment_dist = Counter([s['label'] for s in sentiments])
    avg_sentiment = np.mean([s['sentiment_score'] for s in sentiments])
    
    # Engagement metrics
    total_score = sum(p.score for p in posts)
    total_comments = sum(p.num_comments for p in posts)
    avg_score = total_score / len(posts)
    avg_comments = total_comments / len(posts)
    
    # Subreddit distribution
    subreddit_dist = Counter([p.subreddit for p in posts if p.subreddit])
    
    # Extract common keywords (simple approach)
    all_words = []
    for text in texts:
        words = text.lower().split()
        # Filter out short words and common ones
        words = [w for w in words if len(w) > 4 and w.isalpha()]
        all_words.extend(words)
    
    keyword_counts = Counter(all_words)
    top_keywords = [word for word, count in keyword_counts.most_common(10)]
    
    # Time distribution
    dates = [p.created_at.date() for p in posts]
    date_dist = Counter(dates)
    
    # Most engaging posts
    top_posts = sorted(posts, key=lambda p: p.score, reverse=True)[:5]
    
    return {
        'total_posts': len(posts),
        'sentiment': {
            'positive': sentiment_dist.get('positive', 0),
            'negative': sentiment_dist.get('negative', 0),
            'neutral': sentiment_dist.get('neutral', 0),
            'average_score': float(avg_sentiment),
            'distribution': dict(sentiment_dist)
        },
        'engagement': {
            'total_upvotes': total_score,
            'total_comments': total_comments,
            'avg_upvotes': avg_score,
            'avg_comments': avg_comments
        },
        'subreddits': dict(subreddit_dist.most_common(10)),
        'keywords': top_keywords,
        'date_distribution': {str(k): v for k, v in date_dist.items()},
        'top_posts': [
            {
                'title': p.title,
                'subreddit': p.subreddit,
                'score': p.score,
                'comments': p.num_comments,
                'sentiment': p.sentiment_label,
                'url': p.url,
                'id': p.id
            }
            for p in top_posts
        ]
    }


def generate_query_summary(posts: List[Post], analysis: Dict) -> str:
    """
    Generate AI summary for query results
    
    Args:
        posts: List of posts
        analysis: Analysis results
    
    Returns:
        Summary text
    """
    from pipeline.summary import summarizer
    
    # Build context
    posts_text = []
    for i, post in enumerate(posts[:10], 1):  # Top 10 posts
        sentiment_emoji = "😊" if post.sentiment_score and post.sentiment_score > 0.2 else "😐" if post.sentiment_score and post.sentiment_score > -0.2 else "😟"
        posts_text.append(
            f"{i}. [{sentiment_emoji}] {post.title}\n"
            f"   r/{post.subreddit} | {post.score} upvotes | {post.num_comments} comments"
        )
    
    posts_str = "\n\n".join(posts_text)
    
    # Create prompt
    prompt = f"""You are analyzing Reddit search results. Provide a concise summary.

Query Results: {analysis['total_posts']} posts found

Sentiment Distribution:
- Positive: {analysis['sentiment']['positive']} ({analysis['sentiment']['positive']/analysis['total_posts']*100:.1f}%)
- Negative: {analysis['sentiment']['negative']} ({analysis['sentiment']['negative']/analysis['total_posts']*100:.1f}%)
- Neutral: {analysis['sentiment']['neutral']} ({analysis['sentiment']['neutral']/analysis['total_posts']*100:.1f}%)
- Average Score: {analysis['sentiment']['average_score']:.3f}

Top Keywords: {', '.join(analysis['keywords'][:8])}

Top Subreddits: {', '.join([f"r/{k}" for k in list(analysis['subreddits'].keys())[:5]])}

Sample Posts:
{posts_str}

Provide a 4-5 sentence summary covering:
1. Main themes and topics discussed
2. Overall sentiment and tone
3. Key takeaways or notable points
4. Community engagement level

Summary:"""
    
    try:
        summary = summarizer.generate(prompt, max_tokens=400)
        return summary
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        return "Summary generation unavailable."


# ============================================
# WORKFLOW 2 INTEGRATION
# ============================================

async def process_query_results(post_ids: List[int]) -> Dict:
    """
    Main entry point for Workflow 2 processing
    
    Args:
        post_ids: List of post IDs from query scraping
    
    Returns:
        Complete analysis results
    """
    return await quick_process_posts(post_ids, generate_summary=True)


# ============================================
# TESTING
# ============================================

async def test_quick_pipeline(query: str = "artificial intelligence"):
    """Test the quick pipeline with a query"""
    print(f"🧪 Testing Quick Pipeline with query: '{query}'")
    
    # First, scrape
    from scraper.reddit_scraper import workflow_2_query
    scrape_result = await workflow_2_query(query)
    
    if not scrape_result['success']:
        print("❌ Scraping failed")
        return
    
    # Get post IDs
    async with DatabaseSession() as db:
        result = await db.execute(
            select(Post.id).where(Post.title.contains(query.split()[0]))
            .limit(20)
        )
        post_ids = [row[0] for row in result.all()]
    
    if not post_ids:
        print("❌ No posts found")
        return
    
    # Process
    analysis = await process_query_results(post_ids)
    
    # Print results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    print(f"\nTotal Posts: {analysis['total_posts']}")
    print(f"\nSentiment:")
    print(f"  Positive: {analysis['sentiment']['positive']}")
    print(f"  Negative: {analysis['sentiment']['negative']}")
    print(f"  Neutral: {analysis['sentiment']['neutral']}")
    print(f"  Average: {analysis['sentiment']['average_score']:.3f}")
    print(f"\nTop Keywords: {', '.join(analysis['keywords'][:5])}")
    print(f"\nTop Subreddits: {list(analysis['subreddits'].keys())[:5]}")
    
    if 'summary' in analysis:
        print(f"\n📝 AI Summary:")
        print(f"{analysis['summary']}")
    
    print("\n" + "="*60)


# ============================================
# MAIN
# ============================================

async def main():
    """Run test"""
    await test_quick_pipeline("climate change")


if __name__ == "__main__":
    asyncio.run(main())
