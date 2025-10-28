"""
Sentiment Analysis Pipeline
Uses Cardiff NLP's Twitter-RoBERTa model for sentiment classification
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import asyncio
from sqlalchemy import select, update
from tqdm import tqdm

from backend.db import DatabaseSession
from backend.models import Post
from pipeline.preprocess import preprocess_text

# ============================================
# MODEL SETUP
# ============================================

class SentimentAnalyzer:
    """Singleton wrapper for sentiment analysis model"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Load sentiment analysis model
        
        Available models:
        - cardiffnlp/twitter-roberta-base-sentiment-latest (RECOMMENDED)
        - cardiffnlp/twitter-roberta-base-sentiment
        - distilbert-base-uncased-finetuned-sst-2-english (faster, less accurate)
        """
        if self._model is None:
            print(f"📦 Loading sentiment model: {model_name}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._config = AutoConfig.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded on {self.device}!")
            print(f"   Labels: {self._config.id2label}")
        
        return self._model
    
    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load_model()
        return self._tokenizer
    
    @property
    def config(self):
        if self._config is None:
            self.load_model()
        return self._config


# Global analyzer instance
sentiment_analyzer = SentimentAnalyzer()


# ============================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================

def analyze_sentiment(text: str, preprocess_text_flag: bool = True) -> Dict[str, any]:
    """
    Analyze sentiment of a single text
    
    Args:
        text: Input text
        preprocess_text_flag: Whether to clean text first
    
    Returns:
        Dictionary with:
            - label: 'positive', 'negative', or 'neutral'
            - score: Confidence score (0-1)
            - scores_all: Scores for all labels
            - sentiment_score: Normalized score (-1 to 1)
    """
    # Preprocess if needed
    if preprocess_text_flag:
        text = preprocess_text(text)
    
    # Handle empty text
    if not text or len(text.strip()) < 3:
        return {
            "label": "neutral",
            "score": 0.0,
            "scores_all": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
            "sentiment_score": 0.0
        }
    
    # Tokenize
    tokenizer = sentiment_analyzer.tokenizer
    model = sentiment_analyzer.model
    config = sentiment_analyzer.config
    
    # Truncate long texts
    encoded = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    encoded = {k: v.to(sentiment_analyzer.device) for k, v in encoded.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    # Get scores for each label
    scores = probabilities[0].cpu().numpy()
    
    # Map to labels
    id2label = config.id2label
    scores_dict = {id2label[i]: float(scores[i]) for i in range(len(scores))}
    
    # Get primary label
    predicted_label_id = torch.argmax(probabilities, dim=1).item()
    predicted_label = id2label[predicted_label_id]
    confidence = float(scores[predicted_label_id])
    
    # Calculate normalized sentiment score (-1 to 1)
    # negative: -1, neutral: 0, positive: 1
    if "negative" in scores_dict and "positive" in scores_dict:
        sentiment_score = scores_dict["positive"] - scores_dict["negative"]
    else:
        # Fallback for different label formats
        sentiment_score = 0.0
        if predicted_label in ["positive", "POSITIVE", "pos"]:
            sentiment_score = confidence
        elif predicted_label in ["negative", "NEGATIVE", "neg"]:
            sentiment_score = -confidence
    
    return {
        "label": predicted_label.lower(),
        "score": confidence,
        "scores_all": scores_dict,
        "sentiment_score": sentiment_score
    }


def analyze_sentiment_batch(
    texts: List[str],
    preprocess_texts: bool = True,
    batch_size: int = 16
) -> List[Dict[str, any]]:
    """
    Analyze sentiment for multiple texts (faster than individual)
    
    Args:
        texts: List of input texts
        preprocess_texts: Whether to clean texts first
        batch_size: Batch size for processing
    
    Returns:
        List of sentiment dictionaries
    """
    # Preprocess if needed
    if preprocess_texts:
        texts = [preprocess_text(t) for t in texts]
    
    # Replace empty texts
    texts = [t if t and len(t.strip()) >= 3 else "neutral text" for t in texts]
    
    tokenizer = sentiment_analyzer.tokenizer
    model = sentiment_analyzer.model
    config = sentiment_analyzer.config
    
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encoded = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(sentiment_analyzer.device) for k, v in encoded.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Process each result in batch
        for j, probs in enumerate(probabilities):
            scores = probs.cpu().numpy()
            id2label = config.id2label
            
            scores_dict = {id2label[i]: float(scores[i]) for i in range(len(scores))}
            
            predicted_label_id = torch.argmax(probs).item()
            predicted_label = id2label[predicted_label_id]
            confidence = float(scores[predicted_label_id])
            
            # Calculate sentiment score
            if "negative" in scores_dict and "positive" in scores_dict:
                sentiment_score = scores_dict["positive"] - scores_dict["negative"]
            else:
                sentiment_score = 0.0
            
            results.append({
                "label": predicted_label.lower(),
                "score": confidence,
                "scores_all": scores_dict,
                "sentiment_score": sentiment_score
            })
    
    return results


# ============================================
# DATABASE INTEGRATION
# ============================================

async def analyze_posts_sentiment(limit: Optional[int] = None):
    """
    Analyze sentiment for all posts in database
    
    Args:
        limit: Maximum number of posts to process (None for all)
    """
    print("😊 Starting sentiment analysis...")
    
    async with DatabaseSession() as db:
        # Get posts without sentiment
        query = select(Post).where(
            (Post.sentiment_label == None) | (Post.sentiment_score == None)
        )
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if not posts:
            print("✅ All posts already analyzed!")
            return
        
        print(f"📊 Analyzing {len(posts)} posts...")
        
        # Prepare texts
        texts = []
        for post in posts:
            full_text = f"{post.title} {post.content or ''}"
            texts.append(full_text)
        
        # Analyze in batches with progress bar
        print("⚙️  Processing batches...")
        sentiments = []
        
        batch_size = 16
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch_texts = texts[i:i + batch_size]
            batch_results = analyze_sentiment_batch(batch_texts, preprocess_texts=True, batch_size=batch_size)
            sentiments.extend(batch_results)
        
        # Update database
        print("💾 Saving results to database...")
        for post, sentiment in zip(posts, sentiments):
            post.sentiment_label = sentiment["label"]
            post.sentiment_score = sentiment["sentiment_score"]
        
        await db.commit()
        
        # Print statistics
        positive = sum(1 for s in sentiments if s["label"] == "positive")
        negative = sum(1 for s in sentiments if s["label"] == "negative")
        neutral = sum(1 for s in sentiments if s["label"] == "neutral")
        
        print(f"\n📈 Sentiment Distribution:")
        print(f"   Positive: {positive} ({positive/len(sentiments)*100:.1f}%)")
        print(f"   Negative: {negative} ({negative/len(sentiments)*100:.1f}%)")
        print(f"   Neutral:  {neutral} ({neutral/len(sentiments)*100:.1f}%)")
        
        avg_sentiment = np.mean([s["sentiment_score"] for s in sentiments])
        print(f"   Average Score: {avg_sentiment:.3f}")
        
        print(f"\n✅ Sentiment analysis complete!")


# ============================================
# ANALYTICS & STATISTICS
# ============================================

async def get_sentiment_statistics(hours: int = 24) -> Dict:
    """
    Get sentiment statistics for recent posts
    
    Args:
        hours: Time window in hours
    
    Returns:
        Dictionary with statistics
    """
    from datetime import datetime, timedelta
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    async with DatabaseSession() as db:
        query = select(Post).where(
            Post.created_at >= cutoff_time,
            Post.sentiment_label != None
        )
        
        result = await db.execute(query)
        posts = result.scalars().all()
        
        if not posts:
            return {"error": "No posts found"}
        
        sentiments = [p.sentiment_label for p in posts]
        scores = [p.sentiment_score for p in posts]
        
        stats = {
            "total_posts": len(posts),
            "positive": sentiments.count("positive"),
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral"),
            "avg_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "most_positive": max(posts, key=lambda p: p.sentiment_score or -1),
            "most_negative": min(posts, key=lambda p: p.sentiment_score or 1)
        }
        
        return stats


# ============================================
# TESTING & DEMO
# ============================================

def demo():
    """Demo sentiment analysis on sample texts"""
    sample_texts = [
        "This is absolutely amazing! I love it! 🎉",
        "Terrible news. This is very disappointing and sad.",
        "The weather is okay today. Nothing special.",
        "Breaking: Major technological breakthrough announced!",
        "Another boring day at work.",
    ]
    
    print("🧪 Sentiment Analysis Demo:\n")
    
    for text in sample_texts:
        result = analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"   Label: {result['label']} (confidence: {result['score']:.3f})")
        print(f"   Score: {result['sentiment_score']:.3f}")
        print(f"   All: {result['scores_all']}\n")


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Analyze sentiment for all posts"""
    # Option 1: Run demo
    # demo()
    
    # Option 2: Process database posts
    await analyze_posts_sentiment(limit=None)
    
    # Option 3: Get statistics
    # stats = await get_sentiment_statistics(hours=24)
    # print("\n📊 Recent Sentiment Stats:")
    # print(f"   Total: {stats['total_posts']}")
    # print(f"   Positive: {stats['positive']} ({stats['positive']/stats['total_posts']*100:.1f}%)")
    # print(f"   Average: {stats['avg_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
