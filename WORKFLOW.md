# AI News Assistant - System Workflow

## 🔄 Complete System Workflow

### **Phase 1: Data Collection & Ingestion**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Reddit API    │    │  Google News    │    │   Pushshift     │
│                 │    │                 │    │   (Fallback)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Reddit Scraper  │    │ News Scraper     │    │ Fallback Scraper│
│                 │    │                 │    │                 │
│ • Subreddit     │    │ • Topic Search   │    │ • Historical    │
│   Monitoring    │    │ • RSS Feeds      │    │   Data          │
│ • Query Search  │    │ • API Calls      │    │ • Archive       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Data Ingestion       │
                    │                         │
                    │ • Duplicate Detection   │
                    │ • Content Validation    │
                    │ • Metadata Extraction    │
                    │ • Rate Limiting         │
                    └─────────┬───────────────┘
                              ▼
                    ┌─────────────────────────┐
                    │     SQLite Database     │
                    │                         │
                    │ • Posts Table           │
                    │ • Topics Table          │
                    │ • Trend Scores          │
                    │ • Chat History          │
                    └─────────────────────────┘
```

### **Phase 2: AI Processing Pipeline**

```
┌─────────────────────────┐
│     Raw Posts Data      │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Preprocessing         │
│                         │
│ • Text Cleaning         │
│ • URL Extraction        │
│ • Emoji Handling        │
│ • Language Detection    │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Sentiment Analysis     │
│                         │
│ • RoBERTa Model         │
│ • Score Calculation      │
│ • Label Assignment       │
│ • Confidence Scoring    │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Topic Modeling        │
│                         │
│ • BERTopic Clustering   │
│ • Keyword Extraction    │
│ • Topic Assignment      │
│ • Drift Detection       │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Vector Embeddings      │
│                         │
│ • Sentence Transformers │
│ • FAISS Index Building  │
│ • Similarity Search     │
│ • Semantic Clustering   │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Trend Analysis        │
│                         │
│ • Engagement Scoring    │
│ • Velocity Calculation   │
│ • Momentum Detection     │
│ • Alert Generation      │
└─────────────────────────┘
```

### **Phase 3: User Interaction & Response**

```
┌─────────────────────────┐
│    Frontend Request      │
│                         │
│ • Dashboard Load        │
│ • Chat Message          │
│ • Sentiment Query       │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│    FastAPI Backend      │
│                         │
│ • Request Validation    │
│ • Authentication        │
│ • Rate Limiting         │
│ • CORS Handling         │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Business Logic        │
│                         │
│ • Data Aggregation      │
│ • AI Model Inference    │
│ • Response Generation   │
│ • Caching Strategy      │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Database Queries      │
│                         │
│ • SQLAlchemy ORM        │
│ • Async Operations      │
│ • Connection Pooling    │
│ • Transaction Management│
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   AI Model Pipeline     │
│                         │
│ • Gemini Integration    │
│ • Sentiment Analysis    │
│ • Topic Classification   │
│ • Summary Generation    │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Response Formatting   │
│                         │
│ • JSON Serialization    │
│ • Error Handling        │
│ • Logging & Monitoring  │
│ • Performance Metrics   │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Frontend Response     │
│                         │
│ • React State Update    │
│ • UI Component Render   │
│ • User Interaction      │
│ • Real-time Updates     │
└─────────────────────────┘
```

## 🔧 Component Interactions

### **Backend API Endpoints**

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/api/stats` | GET | Dashboard statistics | None | Stats object |
| `/api/sentiment-distribution/{source}` | GET | Sentiment breakdown | Source name | Distribution data |
| `/api/recent-posts/{source}` | GET | Recent posts | Source, limit | Post array |
| `/api/chat` | POST | Chat interaction | Query, session_id | AI response |
| `/api/analyze-sentiment` | POST | Topic analysis | Topic string | Analysis results |
| `/api/compare-sentiment` | POST | Cross-source comparison | Topic string | Comparison data |
| `/api/refresh` | POST | Data refresh | None | Refresh status |

### **Database Schema**

```sql
-- Posts table
CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    post_id TEXT UNIQUE,
    title TEXT,
    content TEXT,
    url TEXT,
    subreddit TEXT,
    source TEXT,
    score INTEGER,
    num_comments INTEGER,
    sentiment_label TEXT,
    sentiment_score REAL,
    topic_id INTEGER,
    is_trending BOOLEAN,
    created_at TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

-- Topics table
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    name TEXT,
    keywords TEXT,
    num_posts INTEGER,
    avg_sentiment REAL,
    created_at TIMESTAMP
);

-- Trend scores table
CREATE TABLE trend_scores (
    id INTEGER PRIMARY KEY,
    post_id INTEGER,
    engagement_score REAL,
    velocity_score REAL,
    momentum_score REAL,
    total_score REAL,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);
```

### **AI Model Pipeline**

1. **Text Preprocessing**
   - Remove special characters
   - Handle URLs and mentions
   - Normalize text format
   - Language detection

2. **Sentiment Analysis**
   - Load RoBERTa model
   - Tokenize input text
   - Generate sentiment scores
   - Assign labels (positive/negative/neutral)

3. **Topic Modeling**
   - Extract embeddings
   - Apply BERTopic clustering
   - Generate topic keywords
   - Assign topic IDs

4. **Vector Search**
   - Create FAISS index
   - Store embeddings
   - Enable similarity search
   - Support semantic queries

## 🚀 Deployment Workflow

### **Development Environment**
```bash
# 1. Environment setup
cp env.example .env
# Edit .env with your API keys

# 2. Backend setup
pip install -r requirements.txt
python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"

# 3. Frontend setup
cd frontend
npm install

# 4. Launch system
python launcher.py
```

### **Production Deployment**
```bash
# 1. Environment configuration
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export GEMINI_API_KEY="your_key"

# 2. Database setup
python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"

# 3. Backend deployment
gunicorn backend.api:app -w 4 -k uvicorn.workers.UvicornWorker

# 4. Frontend build
cd frontend
npm run build
# Serve with nginx or similar
```

## 📊 Monitoring & Analytics

### **Performance Metrics**
- API response times
- Database query performance
- AI model inference speed
- Memory usage patterns
- Error rates and types

### **Business Metrics**
- User engagement rates
- Chat session duration
- Sentiment accuracy scores
- Topic trend detection
- Data freshness metrics

### **System Health Checks**
- Database connectivity
- API endpoint availability
- AI model loading status
- External service dependencies
- Resource utilization

## 🔒 Security Considerations

### **API Security**
- Rate limiting per IP
- Request validation
- CORS configuration
- Input sanitization
- SQL injection prevention

### **Data Privacy**
- No personal data storage
- Anonymized user sessions
- Secure API key management
- Environment variable protection
- Audit logging

### **Model Security**
- Model versioning
- Input validation
- Output sanitization
- Error handling
- Resource limits

## 🐛 Error Handling & Recovery

### **Common Error Scenarios**
1. **API Rate Limiting**: Implement exponential backoff
2. **Database Connection Loss**: Connection pooling and retry logic
3. **Model Loading Failures**: Fallback to simpler models
4. **Memory Exhaustion**: Garbage collection and resource monitoring
5. **Network Timeouts**: Circuit breaker pattern

### **Recovery Strategies**
- Automatic retry mechanisms
- Graceful degradation
- Fallback data sources
- Health check endpoints
- Alert notifications

## 📈 Scalability Considerations

### **Horizontal Scaling**
- Stateless API design
- Database connection pooling
- Load balancer configuration
- Container orchestration
- Microservice architecture

### **Vertical Scaling**
- Memory optimization
- CPU utilization monitoring
- Database indexing
- Caching strategies
- Resource allocation

---

This workflow ensures a robust, scalable, and maintainable AI news analysis system that can handle real-time data processing and user interactions efficiently.
