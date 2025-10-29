"""
Data Ingestion Utilities
Bulk loading, processing, and data management functions
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from sqlalchemy import select
from tqdm import tqdm

from backend.db import DatabaseSession, init_db
from backend.models import Post, Topic
from pipeline.preprocess import preprocess_text
from pipeline.embeddings import generate_embeddings_batch, embedding_to_list
from pipeline.sentiment import analyze_sentiment_batch

# ============================================
# JSON/CSV LOADING
# ============================================

def load_json_file(filepath: str) -> List[Dict]:
    """
    Load posts from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        List of post dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return []
    
    print(f"📂 Loading {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and single object
    if isinstance(data, dict):
        data = [data]
    
    print(f"✅ Loaded {len(data)} posts")
    return data


def load_csv_file(filepath: str) -> List[Dict]:
    """
    Load posts from CSV file
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        List of post dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return []
    
    print(f"📂 Loading {filepath}...")
    
    posts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        posts = list(reader)
    
    print(f"✅ Loaded {len(posts)} posts")
    return posts


def load_directory(directory: str, file_pattern: str = "*.json") -> List[Dict]:
    """
    Load all matching files from directory
    
    Args:
        directory: Directory path
        file_pattern: Glob pattern for files
    
    Returns:
        Combined list of all posts
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    print(f"📂 Scanning {directory} for {file_pattern}...")
    
    all_posts = []
    files = list(directory.glob(file_pattern))
    
    for filepath in files:
        if filepath.suffix == '.json':
            posts = load_json_file(str(filepath))
        elif filepath.suffix == '.csv':
            posts = load_csv_file(str(filepath))
        else:
            continue
        
        all_posts.extend(posts)
    
    print(f"✅ Total loaded: {len(all_posts)} posts from {len(files)} files")
    return all_posts


# ============================================
# DATABASE INGESTION
# ============================================

async def ingest_posts_to_db(
    posts: List[Dict],
    skip_duplicates: bool = True,
    batch_size: int = 100
) -> Dict[str, int]:
    """
    Ingest posts into database
    
    Args:
        posts: List of post dictionaries
        skip_duplicates: Skip posts that already exist
        batch_size: Number of posts per batch
    
    Returns:
        Dictionary with counts (saved, skipped, errors)
    """
    print(f"\n💾 Ingesting {len(posts)} posts to database...")
    
    saved = 0
    skipped = 0
    errors = 0
    
    async with DatabaseSession() as db:
        for i in tqdm(range(0, len(posts), batch_size), desc="Ingesting batches"):
            batch = posts[i:i + batch_size]
            
            for post_data in batch:
                try:
                    # Check if exists
                    if skip_duplicates:
                        existing = await db.scalar(
                            select(Post).where(Post.post_id == post_data.get('post_id'))
                        )
                        
                        if existing:
                            skipped += 1
                            continue
                    
                    # Parse created_at if string
                    created_at = post_data.get('created_at')
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    
                    # Create post
                    post = Post(
                        post_id=post_data.get('post_id'),
                        source=post_data.get('source', 'reddit'),
                        subreddit=post_data.get('subreddit'),
                        author=post_data.get('author'),
                        title=post_data.get('title'),
                        content=post_data.get('content', ''),
                        url=post_data.get('url'),
                        score=post_data.get('score', 0),
                        num_comments=post_data.get('num_comments', 0),
                        created_at=created_at or datetime.utcnow(),
                        is_processed=False,
                        is_trending=False
                    )
                    
                    db.add(post)
                    saved += 1
                
                except Exception as e:
                    print(f"\n❌ Error ingesting post: {e}")
                    errors += 1
            
            # Commit batch
            await db.commit()
    
    print(f"\n✅ Ingestion complete!")
    print(f"   Saved: {saved}")
    print(f"   Skipped: {skipped}")
    print(f"   Errors: {errors}")
    
    return {
        'saved': saved,
        'skipped': skipped,
        'errors': errors
    }


# ============================================
# FULL PROCESSING PIPELINE
# ============================================

async def process_unprocessed_posts(limit: Optional[int] = None):
    """
    Run full NLP pipeline on unprocessed posts
    
    Args:
        limit: Maximum posts to process (None for all)
    """
    print("\n⚙️  Running full processing pipeline...")
    
    async with DatabaseSession() as db:
        # Get unprocessed posts
        query = select(Post).where(Post.is_processed == False)
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if not posts:
            print("✅ All posts already processed!")
            return
        
        print(f"📊 Processing {len(posts)} posts...\n")
        
        # Step 1: Prepare texts
        print("1️⃣ Preprocessing texts...")
        texts = []
        for post in posts:
            full_text = f"{post.title} {post.content or ''}"
            cleaned = preprocess_text(full_text)
            texts.append(cleaned)
        
        # Step 2: Generate embeddings
        print("2️⃣ Generating embeddings...")
        embeddings = generate_embeddings_batch(texts, preprocess=False, batch_size=32)
        
        for post, embedding in zip(posts, embeddings):
            post.embedding = embedding_to_list(embedding)
        
        await db.commit()
        
        # Step 3: Sentiment analysis
        print("3️⃣ Analyzing sentiment...")
        sentiments = analyze_sentiment_batch(texts, preprocess_texts=False, batch_size=16)
        
        for post, sentiment in zip(posts, sentiments):
            post.sentiment_label = sentiment['label']
            post.sentiment_score = sentiment['sentiment_score']
        
        await db.commit()
        
        # Step 4: Mark as processed
        print("4️⃣ Marking as processed...")
        for post in posts:
            post.is_processed = True
        
        await db.commit()
        
        print(f"\n✅ Processed {len(posts)} posts!")
        
        # Show stats
        positive = sum(1 for s in sentiments if s['label'] == 'positive')
        negative = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral = sum(1 for s in sentiments if s['label'] == 'neutral')
        
        print(f"\n📈 Results:")
        print(f"   Positive: {positive} ({positive/len(sentiments)*100:.1f}%)")
        print(f"   Negative: {negative} ({negative/len(sentiments)*100:.1f}%)")
        print(f"   Neutral:  {neutral} ({neutral/len(sentiments)*100:.1f}%)")


# ============================================
# BULK OPERATIONS
# ============================================

async def delete_all_posts(confirm: bool = False):
    """
    Delete all posts from database
    
    Args:
        confirm: Must be True to actually delete
    """
    if not confirm:
        print("⚠️  Set confirm=True to delete all posts")
        return
    
    print("🗑️  Deleting all posts...")
    
    async with DatabaseSession() as db:
        # Delete all posts
        posts = await db.execute(select(Post))
        posts = posts.scalars().all()
        
        for post in posts:
            await db.delete(post)
        
        await db.commit()
        
        print(f"✅ Deleted {len(posts)} posts")


async def reset_processing_flags():
    """Reset is_processed flag for all posts"""
    print("🔄 Resetting processing flags...")
    
    async with DatabaseSession() as db:
        result = await db.execute(select(Post))
        posts = result.scalars().all()
        
        for post in posts:
            post.is_processed = False
            post.is_trending = False
        
        await db.commit()
        
        print(f"✅ Reset {len(posts)} posts")


# ============================================
# DATA EXPORT
# ============================================

async def export_posts_to_json(
    output_file: str = "data/processed/posts_export.json",
    include_embeddings: bool = False
):
    """
    Export posts to JSON file
    
    Args:
        output_file: Output file path
        include_embeddings: Include embedding vectors (large file)
    """
    print(f"📤 Exporting posts to {output_file}...")
    
    async with DatabaseSession() as db:
        result = await db.execute(select(Post))
        posts = result.scalars().all()
        
        export_data = []
        for post in posts:
            data = {
                'post_id': post.post_id,
                'source': post.source,
                'subreddit': post.subreddit,
                'author': post.author,
                'title': post.title,
                'content': post.content,
                'url': post.url,
                'score': post.score,
                'num_comments': post.num_comments,
                'sentiment_label': post.sentiment_label,
                'sentiment_score': post.sentiment_score,
                'is_trending': post.is_trending,
                'created_at': post.created_at.isoformat() if post.created_at else None
            }
            
            if include_embeddings and post.embedding:
                data['embedding'] = post.embedding
            
            export_data.append(data)
        
        # Write to file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Exported {len(export_data)} posts")


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Main ingestion workflow"""
    print("🚀 Data Ingestion Pipeline\n")
    
    # Initialize database
    await init_db()
    
    # Load data from raw directory
    posts = load_directory("data/raw", "*.json")
    
    if not posts:
        print("\n⚠️  No posts found in data/raw/")
        print("   Run the scraper first: python -m scraper.reddit_scraper")
        return
    
    # Ingest to database
    stats = await ingest_posts_to_db(posts, skip_duplicates=True)
    
    # Process unprocessed posts
    if stats['saved'] > 0:
        print("\n" + "="*60)
        await process_unprocessed_posts(limit=None)


if __name__ == "__main__":
    asyncio.run(main())
