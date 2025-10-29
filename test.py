# update_db.py
import asyncio
from backend.db import init_db

async def main():
    print("🔄 Updating database with new tables...")
    await init_db()
    print("✅ Database updated!")

if __name__ == "__main__":
    asyncio.run(main())
