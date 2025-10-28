"""
Topic Modeling with BERTopic
Discovers and clusters topics from posts
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path
import asyncio
from sqlalchemy import select
from datetime import datetime

from backend.db import DatabaseSession
from backend.models import Post, Topic, TopicDrift
from pipeline.preprocess import preprocess_text
from pipeline.embeddings import embedding_model, list_to_embedding

# ============================================
# BERTOPIC MODEL SETUP
# ============================================

class TopicModelManager:
    """Manages BERTopic model training and inference"""
    
    def __init__(self):
        self.model = None
        self.model_path = Path("models/bertopic/model")
    
    def create_model(
        self,
        language: str = "english",
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None
    ):
        """
        Create BERTopic model with custom parameters
        
        Args:
            language: Language for stopwords
            min_topic_size: Minimum posts per topic
            nr_topics: Target number of topics (None = auto)
        """
        print("🔧 Creating BERTopic model...")
        
        # Use same embedding model as embeddings.py
        embedding_model_instance = SentenceTransformer("all-MiniLM-L6-v2")
        
        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # CountVectorizer for topic words
        vectorizer_model = CountVectorizer(
            stop_words=language,
            min_df=2,
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=embedding_model_instance,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            top_n_words=10,
            language=language,
            calculate_probabilities=True,
            verbose=True
        )
        
        print("✅ BERTopic model created!")
        return self.model
    
    def train(self, texts: List[str], embeddings: Optional[np.ndarray] = None):
        """
        Train BERTopic model on texts
        
        Args:
            texts: List of documents to train on
            embeddings: Pre-computed embeddings (optional)
        """
        print(f"🎓 Training BERTopic on {len(texts)} documents...")
        
        if self.model is None:
            self.create_model()
        
        # Train model
        topics, probabilities = self.model.fit_transform(texts, embeddings)
        
        # Get topic info
        topic_info = self.model.get_topic_info()
        
        print(f"\n✅ Training complete!")
        print(f"   Discovered {len(topic_info) - 1} topics")  # -1 for outlier topic
        print(f"   Outliers: {sum(1 for t in topics if t == -1)}")
        
        return topics, probabilities
    
    def get_topic_info(self) -> Dict:
        """Get information about discovered topics"""
        if self.model is None:
            return {}
        
        topic_info = self.model.get_topic_info()
        
        topics_dict = {}
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue  # Skip outlier topic
            
            # Get topic words
            topic_words = self.model.get_topic(topic_id)
            keywords = [word for word, score in topic_words[:10]]
            
            # Generate topic name from top words
            topic_name = "_".join(keywords[:3])
            
            topics_dict[topic_id] = {
                "id": topic_id,
                "name": topic_name,
                "keywords": keywords,
                "count": row['Count']
            }
        
        return topics_dict
    
    def predict(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Predict topics for new texts
        
        Args:
            texts: List of documents
        
        Returns:
            (topics, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        topics, probabilities = self.model.transform(texts)
        return topics, probabilities
    
    def save(self, path: Optional[str] = None):
        """Save BERTopic model"""
        if path is None:
            path = self.model_path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # BERTopic has built-in save
        self.model.save(str(path), serialization="pickle", save_ctfidf=True)
        
        print(f"💾 Model saved to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load BERTopic model"""
        if path is None:
            path = self.model_path
        
        if not Path(path).exists():
            print(f"⚠️  Model not found: {path}")
            return False
        
        self.model = BERTopic.load(str(path))
        print(f"✅ Model loaded from {path}")
        return True


# Global model manager
topic_manager = TopicModelManager()


# ============================================
# DATABASE INTEGRATION
# ============================================

async def train_topic_model(min_posts: int = 50):
    """
    Train BERTopic model on posts from database
    
    Args:
        min_posts: Minimum number of posts needed to train
    """
    print("🎓 Starting topic modeling training...")
    
    async with DatabaseSession() as db:
        # Get posts with embeddings
        query = select(Post).where(Post.embedding != None)
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if len(posts) < min_posts:
            print(f"⚠️  Need at least {min_posts} posts, found {len(posts)}")
            return
        
        print(f"📊 Training on {len(posts)} posts...")
        
        # Prepare data
        texts = []
        embeddings_list = []
        post_ids = []
        
        for post in posts:
            full_text = f"{post.title} {post.content or ''}"
            cleaned_text = preprocess_text(full_text)
            
            if len(cleaned_text) < 10:
                continue
            
            texts.append(cleaned_text)
            embeddings_list.append(list_to_embedding(post.embedding))
            post_ids.append(post.id)
        
        embeddings = np.array(embeddings_list)
        
        print(f"   Using {len(texts)} valid posts")
        
        # Train model
        topics, probabilities = topic_manager.train(texts, embeddings)
        
        # Save model
        topic_manager.save()
        
        # Save topics to database
        await save_topics_to_database(db, post_ids, topics, texts)
        
        print("✅ Topic modeling complete!")


async def save_topics_to_database(db, post_ids: List[int], topics: List[int], texts: List[str]):
    """Save discovered topics and assignments to database"""
    print("\n💾 Saving topics to database...")
    
    # Get topic info
    topics_info = topic_manager.get_topic_info()
    
    # Create Topic entries
    topic_db_map = {}  # Maps BERTopic topic_id to database Topic.id
    
    for topic_id, info in topics_info.items():
        # Check if topic exists
        existing = await db.scalar(
            select(Topic).where(Topic.topic_num == topic_id)
        )
        
        if existing:
            topic_db = existing
            # Update info
            topic_db.name = info['name']
            topic_db.keywords = info['keywords']
            topic_db.num_posts = info['count']
        else:
            # Create new topic
            topic_db = Topic(
                topic_num=topic_id,
                name=info['name'],
                keywords=info['keywords'],
                num_posts=info['count']
            )
            db.add(topic_db)
        
        await db.flush()
        topic_db_map[topic_id] = topic_db.id
    
    # Assign topics to posts
    print("   Assigning topics to posts...")
    assigned_count = 0
    
    for post_id, topic_num in zip(post_ids, topics):
        if topic_num == -1:  # Outlier
            continue
        
        if topic_num not in topic_db_map:
            continue
        
        # Update post
        result = await db.execute(select(Post).where(Post.id == post_id))
        post = result.scalar_one_or_none()
        
        if post:
            post.topic_id = topic_db_map[topic_num]
            assigned_count += 1
    
    await db.commit()
    
    print(f"✅ Saved {len(topics_info)} topics, assigned {assigned_count} posts")


async def assign_topics_to_new_posts():
    """Assign topics to posts that don't have one yet"""
    print("🏷️  Assigning topics to new posts...")
    
    # Load model
    if not topic_manager.load():
        print("⚠️  Model not found. Train first!")
        return
    
    async with DatabaseSession() as db:
        # Get posts without topics
        query = select(Post).where(
            Post.topic_id == None,
            Post.embedding != None
        )
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if not posts:
            print("✅ All posts already have topics!")
            return
        
        print(f"📊 Processing {len(posts)} posts...")
        
        # Prepare texts
        texts = []
        valid_posts = []
        
        for post in posts:
            full_text = f"{post.title} {post.content or ''}"
            cleaned_text = preprocess_text(full_text)
            
            if len(cleaned_text) >= 10:
                texts.append(cleaned_text)
                valid_posts.append(post)
        
        if not texts:
            print("⚠️  No valid texts found")
            return
        
        # Predict topics
        topics, probabilities = topic_manager.predict(texts)
        
        # Get topic DB mapping
        topic_result = await db.execute(select(Topic))
        all_topics = topic_result.scalars().all()
        topic_map = {t.topic_num: t.id for t in all_topics}
        
        # Assign topics
        assigned = 0
        for post, topic_num in zip(valid_posts, topics):
            if topic_num == -1:  # Outlier
                continue
            
            if topic_num in topic_map:
                post.topic_id = topic_map[topic_num]
                assigned += 1
        
        await db.commit()
        
        print(f"✅ Assigned topics to {assigned} posts")


# ============================================
# TOPIC DRIFT TRACKING
# ============================================

async def create_topic_snapshot():
    """Create a snapshot of current topic state for drift tracking"""
    print("📸 Creating topic snapshot...")
    
    async with DatabaseSession() as db:
        query = select(Topic)
        result = await db.execute(query)
        topics = result.scalars().all()
        
        snapshot_time = datetime.utcnow()
        
        for topic in topics:
            # Get recent posts for this topic
            posts_query = select(Post).where(
                Post.topic_id == topic.id
            ).order_by(Post.created_at.desc()).limit(10)
            
            posts_result = await db.execute(posts_query)
            recent_posts = posts_result.scalars().all()
            
            if not recent_posts:
                continue
            
            # Calculate average sentiment
            sentiments = [p.sentiment_score for p in recent_posts if p.sentiment_score is not None]
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Create drift snapshot
            drift = TopicDrift(
                topic_id=topic.id,
                snapshot_time=snapshot_time,
                keywords=topic.keywords,
                post_count=topic.num_posts,
                avg_sentiment=float(avg_sentiment),
                top_posts=[p.id for p in recent_posts[:5]]
            )
            
            db.add(drift)
        
        await db.commit()
        print(f"✅ Created snapshot for {len(topics)} topics")


# ============================================
# ANALYTICS
# ============================================

async def get_top_topics(limit: int = 10) -> List[Dict]:
    """Get most popular topics"""
    async with DatabaseSession() as db:
        query = select(Topic).order_by(Topic.num_posts.desc()).limit(limit)
        result = await db.execute(query)
        topics = result.scalars().all()
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "keywords": t.keywords,
                "num_posts": t.num_posts,
                "avg_sentiment": t.avg_sentiment
            }
            for t in topics
        ]


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Full topic modeling pipeline"""
    print("🚀 Starting topic modeling pipeline...\n")
    
    # Step 1: Train model
    await train_topic_model(min_posts=50)
    
    # Step 2: Assign topics to any unassigned posts
    await assign_topics_to_new_posts()
    
    # Step 3: Create snapshot for drift tracking
    await create_topic_snapshot()
    
    # Step 4: Show top topics
    print("\n📊 Top Topics:")
    top_topics = await get_top_topics(limit=10)
    for i, topic in enumerate(top_topics, 1):
        print(f"\n{i}. {topic['name']} ({topic['num_posts']} posts)")
        print(f"   Keywords: {', '.join(topic['keywords'][:5])}")
        if topic['avg_sentiment']:
            print(f"   Sentiment: {topic['avg_sentiment']:.3f}")
    
    print("\n✅ Topic modeling pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
