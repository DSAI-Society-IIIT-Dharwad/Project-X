"""
Database Configuration and Session Management
Handles SQLite (hackathon) and PostgreSQL (production) connections
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./news_bot.db")

# Import Base from models to avoid circular imports
from backend.models import Base

# ============================================
# Database Engine Configuration
# ============================================

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "False").lower() == "true",  # Log SQL queries in debug mode
    future=True,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,  # Connection pool size
    max_overflow=20,  # Max connections beyond pool_size
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ============================================
# Database Initialization
# ============================================

async def init_db():
    """
    Initialize database - create all tables
    Call this on application startup
    """
    async with engine.begin() as conn:
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized successfully!")


async def drop_db():
    """
    Drop all tables - USE WITH CAUTION!
    Only for development/testing
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("⚠️  Database dropped!")


# ============================================
# Session Dependency for FastAPI
# ============================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI routes
    Provides database session and handles cleanup
    
    Usage in FastAPI:
        @app.get("/posts")
        async def get_posts(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================
# Utility Functions
# ============================================

async def check_connection():
    """
    Test database connection
    Returns True if successful, False otherwise
    """
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def get_table_counts():
    """
    Get row counts for all tables
    Useful for debugging/monitoring
    """
    from sqlalchemy import select, func
    from backend.models import Post, Topic, TrendScore, TopicDrift, SummaryCache
    
    async with AsyncSessionLocal() as session:
        counts = {}
        
        # Count posts
        result = await session.execute(select(func.count(Post.id)))
        counts['posts'] = result.scalar()
        
        # Count topics
        result = await session.execute(select(func.count(Topic.id)))
        counts['topics'] = result.scalar()
        
        # Count trend scores
        result = await session.execute(select(func.count(TrendScore.id)))
        counts['trend_scores'] = result.scalar()
        
        # Count topic drift records
        result = await session.execute(select(func.count(TopicDrift.id)))
        counts['topic_drift'] = result.scalar()
        
        # Count cached summaries
        result = await session.execute(select(func.count(SummaryCache.id)))
        counts['summary_cache'] = result.scalar()
        
        return counts


# ============================================
# Context Manager (Alternative Usage)
# ============================================

class DatabaseSession:
    """
    Context manager for manual session handling
    
    Usage:
        async with DatabaseSession() as db:
            result = await db.execute(query)
    """
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = AsyncSessionLocal()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        else:
            await self.session.commit()
        await self.session.close()


# ============================================
# Cleanup on Shutdown
# ============================================

async def close_db():
    """
    Close database connections
    Call this on application shutdown
    """
    await engine.dispose()
    print("✅ Database connections closed!")
