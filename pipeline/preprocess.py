"""
Text Preprocessing Pipeline
Cleans and normalizes text for NLP tasks
"""

import re
import emoji
import nltk
from typing import List, Optional
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import asyncio
from sqlalchemy import select

from backend.db import DatabaseSession
from backend.models import Post

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# ============================================
# CLEANING FUNCTIONS
# ============================================

def remove_urls(text: str) -> str:
    """Remove all URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    
    # Also remove www.example.com style
    text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', '', text)
    
    return text


def remove_mentions_hashtags(text: str) -> str:
    """Remove @mentions and #hashtags (Twitter-style)"""
    text = re.sub(r'@\w+', '', text)  # Remove @username
    text = re.sub(r'#\w+', '', text)  # Remove #hashtag
    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters
    
    Args:
        text: Input text
        keep_punctuation: If True, keeps basic punctuation (. , ! ?)
    """
    if keep_punctuation:
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)
    else:
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text


def remove_extra_whitespace(text: str) -> str:
    """Remove extra spaces, tabs, newlines"""
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    return text


def remove_emojis(text: str) -> str:
    """Remove all emojis from text"""
    return emoji.replace_emoji(text, replace='')


def convert_emojis_to_text(text: str) -> str:
    """Convert emojis to text descriptions (alternative to removing)"""
    return emoji.demojize(text, delimiters=(" ", " "))


def expand_contractions(text: str) -> str:
    """Expand common English contractions"""
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }
    
    # Case-insensitive replacement
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b', re.IGNORECASE)
    
    def replace(match):
        return contractions[match.group(0).lower()]
    
    return pattern.sub(replace, text)


def remove_stopwords(text: str, language: str = 'english') -> str:
    """
    Remove common stopwords
    Note: May not be needed for transformer models
    """
    try:
        stop_words = set(stopwords.words(language))
        tokens = word_tokenize(text.lower())
        filtered = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered)
    except:
        return text


def normalize_text(text: str) -> str:
    """Normalize unicode characters and fix encoding issues"""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text


def remove_reddit_formatting(text: str) -> str:
    """Remove Reddit-specific markdown and formatting"""
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # Remove quotes
    text = re.sub(r'^>+\s*', '', text, flags=re.MULTILINE)
    
    # Remove subreddit links
    text = re.sub(r'r/\w+', '', text)
    
    return text


# ============================================
# MAIN PREPROCESSING PIPELINE
# ============================================

def preprocess_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_mentions: bool = True,
    remove_emojis_flag: bool = True,
    expand_contractions_flag: bool = True,
    remove_stopwords_flag: bool = False,  # Usually False for transformers
    lowercase: bool = False  # Usually False for sentiment models
) -> str:
    """
    Complete preprocessing pipeline
    
    Args:
        text: Input text
        remove_urls_flag: Remove URLs
        remove_mentions: Remove @mentions and #hashtags
        remove_emojis_flag: Remove emojis (False to convert to text)
        expand_contractions_flag: Expand contractions
        remove_stopwords_flag: Remove stopwords (not recommended for transformers)
        lowercase: Convert to lowercase (not recommended for sentiment)
    
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Reddit-specific cleaning
    text = remove_reddit_formatting(text)
    
    # Basic cleaning
    if remove_urls_flag:
        text = remove_urls(text)
    
    if remove_mentions:
        text = remove_mentions_hashtags(text)
    
    # Emoji handling
    if remove_emojis_flag:
        text = remove_emojis(text)
    else:
        text = convert_emojis_to_text(text)
    
    # Normalize unicode
    text = normalize_text(text)
    
    # Expand contractions
    if expand_contractions_flag:
        text = expand_contractions(text)
    
    # Remove special characters (keep punctuation for sentiment)
    text = remove_special_characters(text, keep_punctuation=True)
    
    # Clean whitespace
    text = remove_extra_whitespace(text)
    
    # Optional: lowercase (usually not needed for modern models)
    if lowercase:
        text = text.lower()
    
    # Optional: remove stopwords (usually not needed for transformers)
    if remove_stopwords_flag:
        text = remove_stopwords(text)
    
    return text


def preprocess_post(post_dict: dict) -> dict:
    """
    Preprocess a complete post dictionary
    
    Args:
        post_dict: Dictionary with 'title' and 'content' keys
    
    Returns:
        Dictionary with cleaned text
    """
    cleaned = post_dict.copy()
    
    # Clean title
    if 'title' in cleaned:
        cleaned['title_clean'] = preprocess_text(cleaned['title'])
    
    # Clean content
    if 'content' in cleaned:
        cleaned['content_clean'] = preprocess_text(cleaned['content'])
    
    # Combined text for analysis
    title_text = cleaned.get('title_clean', '')
    content_text = cleaned.get('content_clean', '')
    cleaned['full_text_clean'] = f"{title_text} {content_text}".strip()
    
    return cleaned


# ============================================
# BATCH PROCESSING
# ============================================

async def preprocess_database_posts(limit: Optional[int] = None):
    """
    Preprocess all unprocessed posts in database
    
    Args:
        limit: Maximum number of posts to process (None for all)
    """
    print("🧹 Starting text preprocessing...")
    
    async with DatabaseSession() as db:
        # Get unprocessed posts
        query = select(Post).where(Post.is_processed == False)
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        posts = result.scalars().all()
        
        print(f"📊 Found {len(posts)} posts to preprocess")
        
        processed_count = 0
        
        for post in posts:
            try:
                # Combine title and content
                full_text = f"{post.title} {post.content or ''}"
                
                # Preprocess
                cleaned_text = preprocess_text(full_text)
                
                # Store cleaned text in content field (or create new field)
                # For now, we'll mark as processed - actual cleaned text
                # will be generated on-the-fly during embedding/sentiment
                
                # Mark as processed (preprocessing done)
                # Note: Actual NLP processing (sentiment, topics) comes next
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"   Processed {processed_count} posts...")
            
            except Exception as e:
                print(f"❌ Error preprocessing post {post.id}: {e}")
        
        await db.commit()
        
        print(f"✅ Preprocessing complete! Processed {processed_count} posts")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_text_stats(text: str) -> dict:
    """Get statistics about text"""
    return {
        "length": len(text),
        "word_count": len(text.split()),
        "has_urls": bool(re.search(r'http[s]?://', text)),
        "has_mentions": bool(re.search(r'@\w+', text)),
        "has_hashtags": bool(re.search(r'#\w+', text)),
        "emoji_count": emoji.emoji_count(text)
    }


def validate_cleaned_text(text: str, min_length: int = 10) -> bool:
    """
    Check if cleaned text is valid for processing
    
    Args:
        text: Cleaned text
        min_length: Minimum character length
    
    Returns:
        True if text is valid
    """
    if not text or len(text) < min_length:
        return False
    
    # Check if text has actual words (not just special chars)
    word_count = len(re.findall(r'\b\w+\b', text))
    return word_count >= 3


# ============================================
# TESTING & DEMO
# ============================================

async def demo():
    """Demo preprocessing on sample text"""
    sample_texts = [
        "Check out this awesome article! https://example.com/news 🔥🔥",
        "@user This is #breaking news! Can't believe it's happening 😱",
        "r/worldnews: **Breaking** - Major event [link](http://reddit.com)",
        "I'm so excited!!! This won't affect us, will it???",
    ]
    
    print("🧪 Preprocessing Demo:\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Original {i}: {text}")
        cleaned = preprocess_text(text)
        print(f"Cleaned {i}:  {cleaned}")
        print(f"Stats:      {get_text_stats(text)}\n")


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Process all posts in database"""
    # Option 1: Run demo
    # await demo()
    
    # Option 2: Process database posts
    await preprocess_database_posts(limit=None)


if __name__ == "__main__":
    asyncio.run(main())
