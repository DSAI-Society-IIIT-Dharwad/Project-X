"""
Text Embedding Generation
Creates vector embeddings using Sentence Transformers
Used for semantic search and similarity matching
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path
import asyncio
from sqlalchemy import select, update

from backend.db import DatabaseSession
from backend.models import Post
from pipeline.preprocess import preprocess_text

# ============================================
# MODEL SETUP
# ============================================

class EmbeddingModel:
    """Singleton wrapper for Sentence Transformer model"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load sentence transformer model
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dimensions (RECOMMENDED)
        - all-mpnet-base-v2: Better quality, 768 dimensions, slower
        - paraphrase-multilingual: Supports 50+ languages
        """
        if self._model is None:
            print(f"📦 Loading embedding model: {model_name}...")
            self._model = SentenceTransformer(model_name)
            print(f"✅ Model loaded! Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        
        return self._model
    
    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()


# Global model instance
embedding_model = EmbeddingModel()


# ============================================
# EMBEDDING GENERATION
# ============================================

def generate_embedding(text: str, preprocess: bool = True) -> np.ndarray:
    """
    Generate embedding for a single text
    
    Args:
        text: Input text
        preprocess: Whether to clean text first
    
    Returns:
        Numpy array of embeddings
    """
    if preprocess:
        text = preprocess_text(text)
    
    if not text or len(text.strip()) < 5:
        # Return zero vector for empty text
        return np.zeros(embedding_model.dimension)
    
    model = embedding_model.model
    embedding = model.encode(text, convert_to_numpy=True)
    
    return embedding


def generate_embeddings_batch(
    texts: List[str],
    preprocess: bool = True,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for multiple texts (faster)
    
    Args:
        texts: List of input texts
        preprocess: Whether to clean texts first
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        Numpy array of shape (len(texts), embedding_dim)
    """
    if preprocess:
        texts = [preprocess_text(t) for t in texts]
    
    # Replace empty texts with placeholder
    texts = [t if t and len(t.strip()) >= 5 else "empty" for t in texts]
    
    model = embedding_model.model
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    return embeddings


def embedding_to_list(embedding: np.ndarray) -> List[float]:
    """Convert numpy embedding to list for JSON storage"""
    return embedding.tolist()


def list_to_embedding(embedding_list: List[float]) -> np.ndarray:
    """Convert list back to numpy array"""
    return np.array(embedding_list, dtype=np.float32)


# ============================================
# FAISS INDEX MANAGEMENT
# ============================================

class FAISSIndex:
    """Manages FAISS vector index for similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.post_ids = []  # Maps index position to post_id
    
    def create_index(self, use_gpu: bool = False):
        """
        Create new FAISS index
        
        Args:
            use_gpu: Use GPU acceleration (requires faiss-gpu)
        """
        print(f"🔧 Creating FAISS index (dimension={self.dimension})...")
        
        # L2 distance index (can also use IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            print("🚀 Using GPU acceleration")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        print("✅ FAISS index created!")
    
    def add_embeddings(self, embeddings: np.ndarray, post_ids: List[int]):
        """
        Add embeddings to index
        
        Args:
            embeddings: Numpy array of shape (n, dimension)
            post_ids: List of post IDs corresponding to embeddings
        """
        if self.index is None:
            self.create_index()
        
        # Ensure correct dtype
        embeddings = embeddings.astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        self.post_ids.extend(post_ids)
        
        print(f"✅ Added {len(post_ids)} embeddings to index (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
        
        Returns:
            (distances, post_ids) - Lists of distances and corresponding post IDs
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), []
        
        # Ensure correct shape and dtype
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Map indices to post IDs
        result_post_ids = [self.post_ids[idx] for idx in indices[0] if idx < len(self.post_ids)]
        
        return distances[0], result_post_ids
    
    def save(self, index_path: str = "models/faiss_index/index.faiss"):
        """Save FAISS index to disk"""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, index_path)
        
        # Save post_id mapping
        mapping_path = index_path.replace('.faiss', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.post_ids, f)
        
        print(f"💾 FAISS index saved to {index_path}")
    
    def load(self, index_path: str = "models/faiss_index/index.faiss"):
        """Load FAISS index from disk"""
        if not Path(index_path).exists():
            print(f"⚠️  Index not found: {index_path}")
            return False
        
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Load post_id mapping
        mapping_path = index_path.replace('.faiss', '_mapping.pkl')
        with open(mapping_path, 'rb') as f:
            self.post_ids = pickle.load(f)
        
        print(f"✅ FAISS index loaded from {index_path} ({self.index.ntotal} vectors)")
        return True


# ============================================
# DATABASE INTEGRATION
# ============================================

async def generate_embeddings_for_posts(limit: Optional[int] = None):
    """
    Generate and store embeddings for posts in database
    
    Args:
        limit: Maximum number of posts to process (None for all)
    """
    print("🧠 Generating embeddings for posts...")
    
    async with DatabaseSession() as db:
        # Get posts without embeddings
        query = select(Post).where(Post.embedding == None)
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if not posts:
            print("✅ All posts already have embeddings!")
            return
        
        print(f"📊 Processing {len(posts)} posts...")
        
        # Prepare texts
        texts = []
        post_ids = []
        
        for post in posts:
            full_text = f"{post.title} {post.content or ''}"
            texts.append(full_text)
            post_ids.append(post.id)
        
        # Generate embeddings in batch
        print("⚙️  Encoding texts...")
        embeddings = generate_embeddings_batch(texts, preprocess=True, batch_size=32)
        
        # Store embeddings in database
        print("💾 Saving embeddings to database...")
        for post, embedding in zip(posts, embeddings):
            post.embedding = embedding_to_list(embedding)
        
        await db.commit()
        
        print(f"✅ Generated embeddings for {len(posts)} posts!")
        
        return embeddings, post_ids


async def build_faiss_index():
    """
    Build FAISS index from all posts with embeddings
    """
    print("🔨 Building FAISS index...")
    
    async with DatabaseSession() as db:
        # Get all posts with embeddings
        query = select(Post).where(Post.embedding != None)
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if not posts:
            print("⚠️  No posts with embeddings found!")
            return None
        
        print(f"📊 Found {len(posts)} posts with embeddings")
        
        # Extract embeddings and IDs
        embeddings_list = []
        post_ids = []
        
        for post in posts:
            if post.embedding:
                embeddings_list.append(list_to_embedding(post.embedding))
                post_ids.append(post.id)
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype='float32')
        
        # Create and populate FAISS index
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_index.add_embeddings(embeddings, post_ids)
        
        # Save to disk
        faiss_index.save()
        
        print(f"✅ FAISS index built with {len(post_ids)} vectors!")
        
        return faiss_index


# ============================================
# SEMANTIC SEARCH
# ============================================

async def search_similar_posts(query: str, k: int = 10) -> List[dict]:
    """
    Search for posts similar to query
    
    Args:
        query: Search query text
        k: Number of results
    
    Returns:
        List of dicts with post info and similarity scores
    """
    # Load FAISS index
    faiss_index = FAISSIndex()
    if not faiss_index.load():
        print("⚠️  FAISS index not found. Build it first!")
        return []
    
    # Generate query embedding
    query_embedding = generate_embedding(query, preprocess=True)
    
    # Search
    distances, post_ids = faiss_index.search(query_embedding, k=k)
    
    # Fetch posts from database
    async with DatabaseSession() as db:
        results = []
        
        for dist, post_id in zip(distances, post_ids):
            result = await db.execute(select(Post).where(Post.id == post_id))
            post = result.scalar_one_or_none()
            
            if post:
                results.append({
                    "post": post,
                    "distance": float(dist),
                    "similarity": 1 / (1 + float(dist))  # Convert distance to similarity
                })
        
        return results


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Full pipeline: generate embeddings + build FAISS index"""
    print("🚀 Starting embedding pipeline...\n")
    
    # Step 1: Generate embeddings for posts
    await generate_embeddings_for_posts(limit=None)
    
    # Step 2: Build FAISS index
    await build_faiss_index()
    
    # Step 3: Test search
    print("\n🔍 Testing semantic search...")
    results = await search_similar_posts("artificial intelligence news", k=5)
    
    print(f"\n📋 Top {len(results)} similar posts:")
    for i, res in enumerate(results, 1):
        post = res['post']
        similarity = res['similarity']
        print(f"\n{i}. [{similarity:.3f}] {post.title[:80]}...")
    
    print("\n✅ Embedding pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
