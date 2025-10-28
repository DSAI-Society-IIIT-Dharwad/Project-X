import praw
from dotenv import load_dotenv
import os

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

print("Authenticated as:", reddit.user.me())  # should print None for script app, that's ok
for submission in reddit.subreddit("worldnews").hot(limit=3):
    print(submission.title)
