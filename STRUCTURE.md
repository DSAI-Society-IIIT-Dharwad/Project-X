# AI News Assistant - File Structure & Architecture

## 📁 Complete File Structure

```
Project-X/
├── 📁 app/                                    # Application Logic Modules
│   ├── ai_assistant.py                        # AI assistant core logic
│   ├── dashboard.py                           # Dashboard analytics engine
│   └── professional_dashboard.py              # Professional dashboard features
│
├── 📁 backend/                                # FastAPI Backend
│   ├── api.py                                 # Main API (full features)
│   ├── api_simple.py                          # Simplified API version
│   ├── api_mock.py                            # Mock API for testing
│   ├── db.py                                  # Database configuration & session management
│   ├── models.py                              # SQLAlchemy ORM models
│   ├── schemas.py                             # Pydantic request/response schemas
│   ├── ingest.py                              # Data ingestion pipeline
│   └── chat_storage.py                        # Chat history storage & retrieval
│
├── 📁 frontend/                               # React Frontend
│   ├── 📁 src/
│   │   ├── 📁 components/                     # Reusable UI Components
│   │   │   ├── Layout.jsx                     # Main application layout
│   │   │   ├── LoadingSpinner.jsx             # Loading state component
│   │   │   ├── MetricCard.jsx                 # Dashboard metric cards
│   │   │   ├── PostCard.jsx                   # Post display component
│   │   │   ├── LoadingDots.jsx                # Chat loading animation
│   │   │   ├── SentimentResults.jsx           # Sentiment analysis results
│   │   │   └── index.js                       # Component exports
│   │   ├── 📁 pages/                          # Page Components
│   │   │   ├── Dashboard.jsx                  # Main dashboard page
│   │   │   ├── ChatAssistant.jsx              # Chat interface page
│   │   │   └── SentimentAnalysis.jsx         # Sentiment analysis page
│   │   ├── 📁 services/                       # API Services
│   │   │   └── api.js                         # Axios API client
│   │   ├── 📁 store/                          # State Management
│   │   │   └── useStore.js                    # Zustand store configuration
│   │   ├── App.jsx                            # Main React application
│   │   ├── main.jsx                           # Application entry point
│   │   └── index.css                          # Global styles
│   ├── 📁 node_modules/                        # Node.js dependencies
│   ├── package.json                           # Frontend dependencies & scripts
│   ├── package-lock.json                      # Dependency lock file
│   ├── tailwind.config.js                     # Tailwind CSS configuration
│   ├── postcss.config.js                      # PostCSS configuration
│   ├── vite.config.js                         # Vite build configuration
│   └── index.html                             # HTML template
│
├── 📁 pipeline/                               # AI/ML Processing Pipeline
│   ├── embeddings.py                          # Vector embedding generation
│   ├── sentiment.py                           # Sentiment analysis pipeline
│   ├── topic_model.py                         # Topic modeling with BERTopic
│   ├── trend_score.py                         # Trend scoring algorithm
│   ├── topic_drift.py                         # Topic drift detection
│   ├── summary.py                             # Text summarization
│   ├── preprocess.py                          # Data preprocessing utilities
│   └── quick_pipeline.py                      # Quick analysis pipeline
│
├── 📁 scraper/                                # Data Collection Modules
│   ├── reddit_scraper.py                      # Reddit API integration
│   ├── google_news_scraper.py                 # Google News scraping
│   └── push_shift_fallback.py                 # Pushshift fallback scraper
│
├── 📁 models/                                 # AI Models & Data (Runtime Generated)
│   └── 📁 faiss_index/                        # FAISS vector index storage
│
├── 📁 logs/                                   # Application Logs (Runtime Generated)
│
├── 📁 data/                                   # Data Storage (Runtime Generated)
│
├── 📄 config.py                               # Configuration management
├── 📄 requirements.txt                        # Python dependencies
├── 📄 launcher.py                             # System launcher script
├── 📄 setup.py                                # Automated setup script
├── 📄 env.example                             # Environment variables template
├── 📄 README.md                               # Project documentation
├── 📄 WORKFLOW.md                             # System workflow documentation
├── 📄 STRUCTURE.md                            # This file
└── 📄 .env                                    # Environment variables (user created)
```

## 🏗️ Architecture Overview

### **Frontend Architecture (React + Vite)**

```
┌─────────────────────────────────────────────────────────────┐
│                    React Application                        │
├─────────────────────────────────────────────────────────────┤
│  App.jsx                                                    │
│  ├── Router Configuration                                  │
│  ├── Layout Component                                       │
│  └── Page Components                                       │
├─────────────────────────────────────────────────────────────┤
│  Pages/                                                     │
│  ├── Dashboard.jsx          │ ChatAssistant.jsx            │
│  │  ├── Stats Display       │ │  ├── Message History       │
│  │  ├── Charts & Graphs     │ │  ├── Input Interface       │
│  │  └── Real-time Updates   │ │  └── AI Response Display   │
│  └── SentimentAnalysis.jsx  │                             │
│      ├── Topic Input        │                             │
│      ├── Analysis Results   │                             │
│      └── Visualization      │                             │
├─────────────────────────────────────────────────────────────┤
│  Components/                                                │
│  ├── Layout.jsx             │ MetricCard.jsx               │
│  ├── LoadingSpinner.jsx     │ PostCard.jsx                 │
│  ├── LoadingDots.jsx        │ SentimentResults.jsx         │
│  └── index.js (exports)     │                             │
├─────────────────────────────────────────────────────────────┤
│  Services/                                                  │
│  └── api.js (Axios client)                                 │
├─────────────────────────────────────────────────────────────┤
│  Store/                                                     │
│  └── useStore.js (Zustand state management)                │
└─────────────────────────────────────────────────────────────┘
```

### **Backend Architecture (FastAPI)**

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  API Endpoints                                              │
│  ├── /api/stats              │ /api/chat                   │
│  ├── /api/sentiment-dist     │ /api/analyze-sentiment     │
│  ├── /api/recent-posts       │ /api/compare-sentiment     │
│  └── /api/refresh            │ /api/chat-history          │
├─────────────────────────────────────────────────────────────┤
│  Business Logic Layer                                       │
│  ├── Data Aggregation       │ AI Model Integration        │
│  ├── Response Generation     │ Caching Strategy           │
│  └── Error Handling         │ Rate Limiting               │
├─────────────────────────────────────────────────────────────┤
│  Data Access Layer                                          │
│  ├── SQLAlchemy ORM         │ Async Database Sessions     │
│  ├── Connection Pooling     │ Transaction Management      │
│  └── Query Optimization     │ Data Validation            │
├─────────────────────────────────────────────────────────────┤
│  External Integrations                                      │
│  ├── Reddit API (PRAW)      │ Google Gemini API           │
│  ├── Google News Scraper    │ FAISS Vector Search        │
│  └── Pushshift Fallback     │ AI Model Pipeline          │
└─────────────────────────────────────────────────────────────┘
```

### **AI Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Processing Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  Data Input                                                 │
│  ├── Raw Text Posts         │ News Articles               │
│  ├── Metadata Extraction    │ URL Processing              │
│  └── Content Validation     │ Language Detection          │
├─────────────────────────────────────────────────────────────┤
│  Preprocessing Layer                                        │
│  ├── Text Cleaning          │ Tokenization                │
│  ├── URL Handling           │ Special Character Removal   │
│  └── Format Normalization   │ Language Detection          │
├─────────────────────────────────────────────────────────────┤
│  AI Model Layer                                             │
│  ├── Sentiment Analysis     │ Topic Modeling              │
│  │  └── RoBERTa Model       │ │  └── BERTopic Clustering  │
│  ├── Embedding Generation   │ Vector Search              │
│  │  └── Sentence Transformers│ │  └── FAISS Index          │
│  └── Text Summarization     │ Trend Detection            │
│      └── Google Gemini      │ │  └── Custom Algorithms    │
├─────────────────────────────────────────────────────────────┤
│  Output Processing                                          │
│  ├── Score Calculation      │ Label Assignment            │
│  ├── Confidence Scoring     │ Result Formatting           │
│  └── Database Storage       │ Cache Management            │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

### **1. Data Collection Flow**
```
External Sources → Scrapers → Validation → Database
     ↓              ↓           ↓           ↓
Reddit API    → Reddit     → Content    → Posts Table
Google News   → News       → Filtering  → Topics Table
Pushshift     → Fallback   → Dedup      → Trend Scores
```

### **2. Processing Flow**
```
Database → Preprocessing → AI Models → Results → Storage
    ↓           ↓            ↓          ↓         ↓
Raw Posts → Text Clean → Sentiment → Scores → Database
Metadata  → Tokenize  → Topics    → Labels → Cache
URLs      → Validate  → Embeddings→ Vectors→ Index
```

### **3. User Interaction Flow**
```
Frontend → API → Business Logic → Database → AI Pipeline → Response
    ↓       ↓         ↓             ↓           ↓           ↓
React   → FastAPI → Aggregation → SQLite   → Models    → JSON
UI      → Routes  → Validation  → Queries  → Inference → Frontend
State   → CORS   → Rate Limit  → ORM      → Results   → Update
```

## 📊 Database Schema

### **Core Tables**

```sql
-- Posts table (main data storage)
CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT UNIQUE NOT NULL,           -- Reddit post ID
    title TEXT NOT NULL,                    -- Post title
    content TEXT,                           -- Post content/text
    url TEXT,                               -- External URL
    subreddit TEXT,                         -- Subreddit name
    source TEXT NOT NULL,                   -- 'reddit' or 'google_news'
    score INTEGER DEFAULT 0,                -- Reddit score/upvotes
    num_comments INTEGER DEFAULT 0,         -- Number of comments
    sentiment_label TEXT,                   -- 'positive', 'negative', 'neutral'
    sentiment_score REAL,                   -- Sentiment score (-1 to 1)
    topic_id INTEGER,                       -- Associated topic ID
    is_trending BOOLEAN DEFAULT FALSE,      -- Trending flag
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

-- Topics table (topic modeling results)
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                     -- Topic name
    keywords TEXT,                          -- Comma-separated keywords
    num_posts INTEGER DEFAULT 0,           -- Number of posts in topic
    avg_sentiment REAL,                     -- Average sentiment score
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trend scores table (trending analysis)
CREATE TABLE trend_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    engagement_score REAL DEFAULT 0,        -- Engagement-based score
    velocity_score REAL DEFAULT 0,          -- Velocity-based score
    momentum_score REAL DEFAULT 0,         -- Momentum-based score
    total_score REAL DEFAULT 0,            -- Combined trend score
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

-- Chat history table (user interactions)
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,               -- User session ID
    user_query TEXT NOT NULL,              -- User's question
    bot_response TEXT NOT NULL,            -- AI response
    relevant_post_ids TEXT,                -- Comma-separated post IDs
    confidence_score REAL,                 -- Response confidence
    response_time_ms INTEGER,              -- Response time in milliseconds
    query_type TEXT,                       -- Type of query
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Topic drift table (topic evolution tracking)
CREATE TABLE topic_drift (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    drift_score REAL,                       -- Drift measurement
    keyword_changes TEXT,                   -- Changed keywords
    sentiment_shift REAL,                   -- Sentiment shift
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

-- Summary cache table (cached AI summaries)
CREATE TABLE summary_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,      -- Hash of input content
    summary_text TEXT NOT NULL,            -- Generated summary
    model_used TEXT,                        -- Model that generated summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Indexes for Performance**

```sql
-- Performance indexes
CREATE INDEX idx_posts_source ON posts(source);
CREATE INDEX idx_posts_subreddit ON posts(subreddit);
CREATE INDEX idx_posts_created_at ON posts(created_at);
CREATE INDEX idx_posts_sentiment ON posts(sentiment_label);
CREATE INDEX idx_posts_trending ON posts(is_trending);
CREATE INDEX idx_posts_topic ON posts(topic_id);

CREATE INDEX idx_chat_session ON chat_history(session_id);
CREATE INDEX idx_chat_created_at ON chat_history(created_at);

CREATE INDEX idx_trend_post ON trend_scores(post_id);
CREATE INDEX idx_trend_score ON trend_scores(total_score);
```

## 🔧 Configuration Files

### **Environment Variables (.env)**
```bash
# API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=your_user_agent
GEMINI_API_KEY=your_gemini_api_key

# AI Models
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOPIC_MODEL_NAME=bertopic_model

# Database
DATABASE_URL=sqlite+aiosqlite:///./news_bot.db

# Scraping
SCRAPE_LIMIT=100
SUBREDDITS=news,worldnews,politics,technology
KEYWORDS=breaking,trending,urgent,developing

# Server
HOST=0.0.0.0
PORT=8000
FRONTEND_PORT=3000
```

### **Frontend Configuration**

#### **package.json**
```json
{
  "name": "project-x-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "axios": "^1.6.2",
    "framer-motion": "^10.16.16",
    "recharts": "^2.10.3",
    "zustand": "^4.4.7",
    "lucide-react": "^0.294.0",
    "react-hot-toast": "^2.4.1",
    "uuid": "^13.0.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.8",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32"
  }
}
```

#### **tailwind.config.js**
```javascript
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: { /* Custom color palette */ },
        dark: { /* Dark theme colors */ }
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-dark': 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)'
      }
    }
  }
}
```

## 🚀 Deployment Architecture

### **Development Environment**
```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │
│   (Vite Dev)    │◄──►│   (Uvicorn)     │
│   Port: 3000    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┼─────────────────┐
                                 ▼                 │
                        ┌─────────────────┐       │
                        │   SQLite DB     │       │
                        │   (Local)       │       │
                        └─────────────────┘       │
                                 │                 │
                                 ▼                 │
                        ┌─────────────────┐       │
                        │  External APIs  │       │
                        │  • Reddit       │       │
                        │  • Google News  │       │
                        │  • Gemini       │       │
                        └─────────────────┘       │
                                 │                 │
                                 ▼                 │
                        ┌─────────────────┐       │
                        │  AI Models      │       │
                        │  • RoBERTa      │       │
                        │  • BERTopic     │       │
                        │  • FAISS        │       │
                        └─────────────────┘       │
                                                   │
                        ┌─────────────────────────┘
                        ▼
               ┌─────────────────┐
               │   Launcher      │
               │   Script        │
               │   (Orchestrator)│
               └─────────────────┘
```

### **Production Environment**
```
┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   Gunicorn      │
│   (Load Balancer)│◄──►│   (WSGI Server) │
│   Port: 80/443  │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Static Files  │    │   FastAPI App   │
│   (React Build) │    │   (Multiple     │
│                 │    │    Workers)     │
└─────────────────┘    └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        │   (Production   │
                        │    Database)    │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Redis Cache   │
                        │   (Session &    │
                        │    Caching)     │
                        └─────────────────┘
```

## 📈 Monitoring & Logging

### **Log Files Structure**
```
logs/
├── app.log                    # Application logs
├── api.log                    # API request logs
├── error.log                  # Error logs
├── ai_pipeline.log            # AI processing logs
├── scraper.log                # Data scraping logs
└── performance.log             # Performance metrics
```

### **Monitoring Endpoints**
- `/health` - System health check
- `/metrics` - Performance metrics
- `/status` - Component status
- `/logs` - Log access endpoint

---

This comprehensive file structure and architecture documentation provides a complete overview of the AI News Assistant system, enabling developers to understand, maintain, and extend the application effectively.
