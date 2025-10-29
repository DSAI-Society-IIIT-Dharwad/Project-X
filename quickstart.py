"""
Quick Start - Set up NewsBot from scratch
"""

import subprocess
import sys
import time

def run_step(cmd, description):
    """Run a setup step"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              🚀  NEWSBOT QUICK START  🚀                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

This will:
1. Initialize database
2. Scrape Reddit
3. Process data
4. Build models
5. Launch dashboard

⏱️  Estimated time: 5-10 minutes

""")
    
    proceed = input("Continue? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("❌ Cancelled")
        return
    
    steps = [
        ('python -c "import asyncio; from backend.db import init_db; asyncio.run(init_db())"',
         "Step 1/5: Initialize Database"),
        
        ('python -m scraper.reddit_scraper',
         "Step 2/5: Scrape Reddit (this may take a few minutes)"),
        
        ('python -m backend.ingest',
         "Step 3/5: Process & Analyze Posts"),
        
        ('python -m pipeline.embeddings',
         "Step 4/5: Build Search Index"),
        
        ('python -m pipeline.topic_model',
         "Step 5/5: Train Topic Model"),
    ]
    
    for cmd, desc in steps:
        success = run_step(cmd, desc)
        if not success:
            print("\n❌ Quick start failed!")
            return
        time.sleep(1)
    
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              ✅  SETUP COMPLETE!  ✅                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

🎉 NewsBot is ready to use!

To launch the dashboard:
    streamlit run app/realtime_dashboard.py

To launch query analyzer:
    streamlit run app/query_analyzer.py

To launch chatbot:
    python -m app.chatbot

Or use the launcher:
    python launcher.py

""")

if __name__ == "__main__":
    main()
