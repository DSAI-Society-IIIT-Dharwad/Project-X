"""
NewsBot Unified Launcher
Easy management of all components
"""

import subprocess
import sys
import os
from pathlib import Path
import time

BANNER = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              🤖  NEWSBOT LAUNCHER  🤖                    ║
║         Real-Time News Analysis Platform                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""

MENU = """
📋 MAIN MENU:

1️⃣  Launch Real-Time Dashboard (Workflow 1)
2️⃣  Launch Query Analyzer (Workflow 2)
3️⃣  Launch Chatbot (CLI)
4️⃣  Launch API Server
5️⃣  Launch ALL (Dashboard + API)

🔧 SETUP & MAINTENANCE:

6️⃣  Initialize Database
7️⃣  Run Scraper (Monitored Subreddits)
8️⃣  Run Full Pipeline
9️⃣  Clear Cache & Reset

🧪 TESTING:

T️⃣  Test Query Search
S️⃣  System Status

Q️⃣  Quit

"""

def run_command(cmd, description, wait=False):
    """Run a command"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}\n")
    
    if wait:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    else:
        subprocess.Popen(cmd, shell=True)
        return True

def launch_dashboard():
    """Launch real-time dashboard"""
    print("\n🚀 Launching Real-Time Dashboard...")
    print("   URL will be: http://localhost:8501")
    time.sleep(1)
    run_command("streamlit run app/dashboard.py", "Real-Time Dashboard")

def launch_query_analyzer():
    """Launch query analyzer"""
    print("\n🚀 Launching Query Analyzer...")
    print("   URL will be: http://localhost:8501")
    time.sleep(1)
    run_command("streamlit run app/query_analyzer.py", "Query Analyzer")

def launch_chatbot():
    """Launch chatbot CLI"""
    print("\n🚀 Launching Chatbot...")
    time.sleep(1)
    run_command("python -m app.chatbot", "Chatbot CLI", wait=True)

def launch_api():
    """Launch API server"""
    print("\n🚀 Launching API Server...")
    print("   API Docs: http://localhost:8000/docs")
    time.sleep(1)
    run_command("uvicorn backend.app:app --reload --port 8000", "API Server")

def launch_all():
    """Launch dashboard + API"""
    print("\n🚀 Launching Dashboard + API...")
    print("   Dashboard: http://localhost:8501")
    print("   API Docs:  http://localhost:8000/docs")
    time.sleep(1)
    run_command("uvicorn backend.app:app --reload --port 8000", "API Server")
    time.sleep(2)
    run_command("streamlit run app/dashboard.py", "Dashboard")

def init_database():
    """Initialize database"""
    run_command(
        'python -c "import asyncio; from backend.db import init_db; asyncio.run(init_db())"',
        "Initializing Database",
        wait=True
    )

def run_scraper():
    """Run scraper"""
    run_command("python -m scraper.reddit_scraper", "Running Scraper", wait=True)

def run_full_pipeline():
    """Run complete pipeline"""
    steps = [
        ("python -m scraper.reddit_scraper", "Scraping Reddit"),
        ("python -m backend.ingest", "Ingesting & Processing"),
        ("python -m pipeline.embeddings", "Building FAISS Index"),
        ("python -m pipeline.topic_model", "Topic Modeling"),
        ("python -m pipeline.trend_score", "Calculating Trends"),
    ]
    
    for cmd, desc in steps:
        success = run_command(cmd, desc, wait=True)
        if not success:
            print(f"\n❌ Failed: {desc}")
            return
        time.sleep(1)
    
    print("\n✅ Full pipeline complete!")

def clear_cache():
    """Clear cache and reset"""
    print("\n🗑️  Clearing cache...")
    
    try:
        # Delete database
        if Path("news_bot.db").exists():
            os.remove("news_bot.db")
            print("✅ Deleted database")
        
        # Clear models
        import shutil
        if Path("models/bertopic").exists():
            shutil.rmtree("models/bertopic")
            print("✅ Cleared BERTopic models")
        
        if Path("models/faiss_index").exists():
            shutil.rmtree("models/faiss_index")
            print("✅ Cleared FAISS index")
        
        print("\n✅ Cache cleared! Run 'Initialize Database' next.")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def test_query():
    """Test query search"""
    query = input("\n🔍 Enter test query: ")
    if query:
        run_command(
            f'python -m scraper.reddit_scraper "{query}"',
            f"Testing Query: {query}",
            wait=True
        )

def system_status():
    """Check system status"""
    print("\n📊 SYSTEM STATUS")
    print("="*60)
    
    # Check database
    if Path("news_bot.db").exists():
        print("✅ Database exists")
        
        # Count records
        try:
            import asyncio
            from sqlalchemy import select, func
            from backend.db import DatabaseSession
            from backend.models import Post, Topic
            
            async def get_counts():
                async with DatabaseSession() as db:
                    posts = await db.scalar(select(func.count(Post.id)))
                    topics = await db.scalar(select(func.count(Topic.id)))
                    return posts, topics
            
            posts, topics = asyncio.run(get_counts())
            print(f"   📰 Posts: {posts}")
            print(f"   🏷️  Topics: {topics}")
        except:
            print("   ⚠️  Could not read database")
    else:
        print("❌ Database not found")
    
    # Check models
    if Path("models/faiss_index/index.faiss").exists():
        print("✅ FAISS index exists")
    else:
        print("❌ FAISS index not found")
    
    if Path("models/bertopic/model").exists():
        print("✅ BERTopic model exists")
    else:
        print("❌ BERTopic model not found")
    
    # Check config
    if Path("config.py").exists():
        print("✅ Configuration file exists")
    else:
        print("❌ Configuration file missing")
    
    # Check .env
    if Path(".env").exists():
        print("✅ Environment file exists")
    else:
        print("❌ Environment file missing")
    
    print("="*60)

def main():
    """Main launcher"""
    print(BANNER)
    
    while True:
        print(MENU)
        choice = input("Enter your choice: ").strip().upper()
        
        if choice == '1':
            launch_dashboard()
        elif choice == '2':
            launch_query_analyzer()
        elif choice == '3':
            launch_chatbot()
        elif choice == '4':
            launch_api()
        elif choice == '5':
            launch_all()
        elif choice == '6':
            init_database()
        elif choice == '7':
            run_scraper()
        elif choice == '8':
            run_full_pipeline()
        elif choice == '9':
            clear_cache()
        elif choice == 'T':
            test_query()
        elif choice == 'S':
            system_status()
        elif choice == 'Q':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice")
        
        if choice not in ['Q', '1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
