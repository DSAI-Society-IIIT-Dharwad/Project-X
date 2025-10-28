"""
AI Chatbot with Retrieval-Augmented Generation (RAG)
Uses FAISS for semantic search + Gemini for responses
"""

import google.generativeai as genai
import os
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from pipeline.embeddings import (
    generate_embedding,
    FAISSIndex,
    embedding_model
)
from pipeline.summary import GeminiSummarizer
from backend.db import DatabaseSession
from backend.models import Post, Topic
from sqlalchemy import select

# Load environment
load_dotenv()

# ============================================
# CHATBOT CLASS
# ============================================

class NewsBot:
    """AI chatbot with news knowledge"""
    
    def __init__(self):
        self.faiss_index = None
        self.gemini = GeminiSummarizer()
        self.conversation_history = []
        self.max_history = 5
    
    def initialize(self):
        """Load FAISS index and models"""
        print("🤖 Initializing NewsBot...")
        
        # Load FAISS index
        self.faiss_index = FAISSIndex()
        if not self.faiss_index.load():
            print("⚠️  FAISS index not found. Building it...")
            # You need to run embeddings pipeline first
            return False
        
        # Initialize Gemini
        self.gemini.initialize()
        
        # Load embedding model
        _ = embedding_model.model
        
        print("✅ NewsBot ready!")
        return True
    
    async def search_relevant_posts(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for posts relevant to query
        
        Args:
            query: User query
            k: Number of results
        
        Returns:
            List of relevant posts with metadata
        """
        # Generate query embedding
        query_embedding = generate_embedding(query, preprocess=True)
        
        # Search FAISS index
        distances, post_ids = self.faiss_index.search(query_embedding, k=k)
        
        # Fetch posts from database
        async with DatabaseSession() as db:
            relevant_posts = []
            
            for dist, post_id in zip(distances, post_ids):
                result = await db.execute(
                    select(Post).where(Post.id == post_id)
                )
                post = result.scalar_one_or_none()
                
                if post:
                    # Get topic if exists
                    topic_name = None
                    if post.topic_id:
                        topic_result = await db.execute(
                            select(Topic).where(Topic.id == post.topic_id)
                        )
                        topic = topic_result.scalar_one_or_none()
                        if topic:
                            topic_name = topic.name
                    
                    relevant_posts.append({
                        'post': post,
                        'distance': float(dist),
                        'similarity': 1 / (1 + float(dist)),
                        'topic': topic_name
                    })
            
            return relevant_posts
    
    def format_context(self, relevant_posts: List[Dict]) -> str:
        """Format posts as context for LLM"""
        context_parts = []
        
        for i, item in enumerate(relevant_posts, 1):
            post = item['post']
            topic = item['topic'] or "General"
            sentiment = "😊" if post.sentiment_score and post.sentiment_score > 0.2 else "😐" if post.sentiment_score and post.sentiment_score > -0.2 else "😟"
            
            context_parts.append(
                f"[{i}] Topic: {topic} {sentiment}\n"
                f"    Title: {post.title}\n"
                f"    Source: r/{post.subreddit} | Score: {post.score} | Comments: {post.num_comments}\n"
                f"    Date: {post.created_at.strftime('%Y-%m-%d')}\n"
            )
            
            if post.content and len(post.content) > 0:
                content_preview = post.content[:200] + "..." if len(post.content) > 200 else post.content
                context_parts.append(f"    Content: {content_preview}\n")
        
        return "\n".join(context_parts)
    
    def create_prompt(
        self,
        user_query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for Gemini"""
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"User: {exchange['user']}\n"
                history_text += f"Bot: {exchange['bot']}\n\n"
        
        # Build history section separately
        history_section = ""
        if history_text:
            history_section = f"Previous conversation:\n{history_text}\n"
        
        prompt = f"""You are NewsBot, an AI assistant that helps users understand current news trends and topics from Reddit.

    You have access to recent news posts and their context. Answer the user's question based on this information.

    {history_section}Relevant News Posts:
    {context}

    User Question: {user_query}

    Instructions:
    1. Answer based on the provided news posts
    2. Be concise and informative (3-5 sentences)
    3. Mention specific posts or topics when relevant
    4. If sentiment is notable, mention it
    5. If the question isn't related to the provided posts, politely say you don't have that information

    Answer:"""
        
        return prompt

    async def chat(self, user_query: str) -> Dict:
        """
        Process user query and generate response
        
        Args:
            user_query: User's question
        
        Returns:
            Dict with response, sources, and metadata
        """
        print(f"\n💬 User: {user_query}")
        
        # Search for relevant posts
        print("🔍 Searching for relevant posts...")
        relevant_posts = await self.search_relevant_posts(user_query, k=5)
        
        if not relevant_posts:
            return {
                'response': "I couldn't find any relevant posts about that topic. Try asking about recent news in technology, AI, climate, or other trending topics.",
                'sources': [],
                'relevant_posts': [],
                'confidence': 0.0
            }
        
        # Format context
        context = self.format_context(relevant_posts)
        
        # Create prompt
        prompt = self.create_prompt(
            user_query,
            context,
            self.conversation_history
        )
        
        # Generate response
        print("🧠 Generating response...")
        response = self.gemini.generate(prompt, max_tokens=400)
        
        # Calculate confidence based on similarity scores
        avg_similarity = sum(p['similarity'] for p in relevant_posts) / len(relevant_posts)
        confidence = min(avg_similarity * 1.2, 1.0)  # Scale up slightly
        
        # Store in conversation history
        self.conversation_history.append({
            'user': user_query,
            'bot': response,
            'timestamp': datetime.utcnow()
        })
        
        # Keep history limited
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Extract sources
        sources = []
        for item in relevant_posts[:3]:  # Top 3 sources
            post = item['post']
            if post.url:
                sources.append(post.url)
            else:
                sources.append(f"https://reddit.com/r/{post.subreddit}")
        
        print(f"🤖 Bot: {response}\n")
        
        return {
            'response': response,
            'sources': sources,
            'relevant_posts': relevant_posts,
            'confidence': confidence
        }
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🔄 Conversation reset")


# ============================================
# INTERACTIVE CLI
# ============================================

async def interactive_mode():
    """Run chatbot in interactive mode"""
    print("=" * 60)
    print("🤖 NewsBot - AI News Assistant")
    print("=" * 60)
    print("\nCommands:")
    print("  • Type your question to chat")
    print("  • 'reset' - Clear conversation history")
    print("  • 'quit' or 'exit' - Exit chatbot")
    print("=" * 60)
    
    # Initialize bot
    bot = NewsBot()
    if not bot.initialize():
        print("\n❌ Failed to initialize. Please run embedding pipeline first:")
        print("   python -m pipeline.embeddings")
        return
    
    print("\n✨ Bot is ready! Ask me about recent news.\n")
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                bot.reset_conversation()
                continue
            
            # Process query
            result = await bot.chat(user_input)
            
            print(f"\n🤖 Bot: {result['response']}")
            
            # Show confidence
            confidence_emoji = "🎯" if result['confidence'] > 0.7 else "🤔" if result['confidence'] > 0.4 else "🤷"
            print(f"\n{confidence_emoji} Confidence: {result['confidence']:.1%}")
            
            # Show sources
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"   {i}. {source[:70]}...")
            
            print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================
# EXAMPLE QUERIES
# ============================================

async def demo_queries():
    """Demo with example queries"""
    bot = NewsBot()
    if not bot.initialize():
        print("❌ Please build FAISS index first")
        return
    
    example_queries = [
        "What's the latest news about artificial intelligence?",
        "Tell me about climate change discussions",
        "What are people saying about technology startups?",
        "Any trending topics in science?",
        "What's the general sentiment about AI?"
    ]
    
    print("🧪 Running demo queries...\n")
    
    for query in example_queries:
        result = await bot.chat(query)
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Sources: {len(result['sources'])}")
        print("-" * 60)
        await asyncio.sleep(1)  # Rate limiting


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        await demo_queries()
    else:
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
