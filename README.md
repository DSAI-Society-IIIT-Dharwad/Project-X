# InQuiro - Project X

A comprehensive AI-powered news analysis platform that combines Reddit discussions with Google News coverage, providing real-time sentiment analysis, trend detection, and intelligent chat capabilities.

## 🚀 Features

### 📊 **Unified Dashboard**
- Real-time statistics from Reddit and Google News
- Sentiment comparison between sources
- Trending topics and engagement metrics
- Interactive charts and visualizations

### 💬 **AI Chat Assistant**
- Intelligent Q&A based on recent Reddit discussions
- Real-time search and analysis
- Confidence scoring and source attribution
- Session-based conversation history

### 📈 **Sentiment Analysis**
- Topic-based sentiment analysis across Reddit
- Subreddit-specific sentiment breakdowns
- Temporal sentiment trends
- Keyword extraction and analysis

### 🔄 **Real-time Data Processing**
- Automated Reddit scraping
- Google News integration
- Sentiment scoring with AI models
- Topic modeling and clustering

## 🏗️ Architecture

### **Frontend (React + Vite)**
- Modern React 18 with hooks
- Tailwind CSS for styling
- Framer Motion for animations
- Recharts for data visualization
- Zustand for state management

### **Backend (FastAPI + Python)**
- Async FastAPI for high performance
- SQLAlchemy with async SQLite
- PRAW for Reddit API integration
- Google Gemini for AI analysis
- FAISS for vector similarity search

### **AI Pipeline**
- RoBERTa for sentiment analysis
- Sentence Transformers for embeddings
- BERTopic for topic modeling
- Custom ML pipelines for trend detection

## 📁 Project Structure

```
Project-X/
├── 📁 app/                          # Application modules
│   ├── ai_assistant.py              # AI assistant logic
│   ├── dashboard.py                 # Dashboard analytics
│   └── professional_dashboard.py     # Professional dashboard
├── 📁 backend/                      # Backend API
│   ├── api.py                       # Main API (full features)
│   ├── api_simple.py                # Simplified API
│   ├── api_mock.py                  # Mock API for testing
│   ├── db.py                        # Database configuration
│   ├── models.py                    # SQLAlchemy models
│   ├── schemas.py                   # Pydantic schemas
│   ├── ingest.py                    # Data ingestion
│   └── chat_storage.py              # Chat history storage
├── 📁 frontend/                     # React frontend
│   ├── src/
│   │   ├── components/              # Reusable components
│   │   │   ├── Layout.jsx           # Main layout
│   │   │   ├── LoadingSpinner.jsx   # Loading states
│   │   │   ├── MetricCard.jsx       # Dashboard metrics
│   │   │   ├── PostCard.jsx         # Post display
│   │   │   ├── LoadingDots.jsx      # Chat loading
│   │   │   └── SentimentResults.jsx # Analysis results
│   │   ├── pages/                   # Page components
│   │   │   ├── Dashboard.jsx        # Main dashboard
│   │   │   ├── ChatAssistant.jsx    # Chat interface
│   │   │   └── SentimentAnalysis.jsx # Sentiment page
│   │   ├── services/                # API services
│   │   │   └── api.js               # API client
│   │   └── store/                   # State management
│   │       └── useStore.js          # Zustand store
│   ├── package.json                 # Frontend dependencies
│   └── tailwind.config.js          # Tailwind configuration
├── 📁 pipeline/                     # ML Pipeline
│   ├── embeddings.py                # Vector embeddings
│   ├── sentiment.py                 # Sentiment analysis
│   ├── topic_model.py               # Topic modeling
│   ├── trend_score.py               # Trend scoring
│   ├── topic_drift.py               # Topic drift detection
│   ├── summary.py                   # Text summarization
│   ├── preprocess.py                # Data preprocessing
│   └── quick_pipeline.py            # Quick analysis pipeline
├── 📁 scraper/                      # Data Scrapers
│   ├── reddit_scraper.py            # Reddit data collection
│   ├── google_news_scraper.py       # Google News scraping
│   └── push_shift_fallback.py       # Pushshift fallback
├── 📁 models/                       # AI Models (created at runtime)
│   └── faiss_index/                 # FAISS vector index
├── 📄 config.py                     # Configuration management
├── 📄 requirements.txt              # Python dependencies
├── 📄 launcher.py                   # System launcher
├── 📄 env.example                   # Environment template
└── 📄 README.md                     # This file
```

## 🔄 System Workflow

### **1. Data Collection**
```
Reddit API → Reddit Scraper → Database
Google News → News Scraper → Database
```

### **2. Data Processing**
```
Raw Posts → Preprocessing → Sentiment Analysis → Topic Modeling → Vector Embeddings
```

### **3. Analysis Pipeline**
```
Posts → FAISS Index → Similarity Search → Trend Detection → Dashboard Updates
```

### **4. User Interaction**
```
Frontend Request → API Endpoint → Database Query → AI Analysis → Response
```

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+
- Node.js 16+
- npm or yarn
- Reddit API credentials
- Google Gemini API key

### **1. Clone Repository**
```bash
git clone <repository-url>
cd Project-X
```

### **2. Environment Setup**
```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env
```

### **3. Backend Setup**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Initialize database
cd backend
python -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from backend.db import init_db
asyncio.run(init_db())
print('Database initialized!')
"
```

### **4. Frontend Setup**
```bash
# Install Node.js dependencies
cd frontend
npm install
```

### **5. Launch System**
```bash
# From project root
python launcher.py
```

## 🌐 Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

## 🔧 Configuration

### **Environment Variables**
Key configuration options in `.env`:

```bash
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Google Gemini
GEMINI_API_KEY=your_gemini_key

# Database
DATABASE_URL=sqlite+aiosqlite:///./news_bot.db

# Scraping
SCRAPE_LIMIT=100
SUBREDDITS=news,worldnews,politics,technology
```

### **Model Configuration**
- **Sentiment Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Topic Model**: BERTopic with custom configuration

## 📊 API Endpoints

### **Dashboard**
- `GET /api/stats` - Overall statistics
- `GET /api/sentiment-distribution/{source}` - Sentiment breakdown
- `GET /api/recent-posts/{source}` - Recent posts

### **Chat**
- `POST /api/chat` - Send chat message
- `GET /api/chat-history/{session_id}` - Get chat history

### **Analysis**
- `POST /api/analyze-sentiment` - Topic sentiment analysis
- `POST /api/compare-sentiment` - Cross-source comparison
- `POST /api/refresh` - Refresh data

## 🚀 Development

### **Running in Development Mode**
```bash
# Backend with hot reload
cd backend
python -m uvicorn api_mock:app --reload --host 0.0.0.0 --port 8000

# Frontend with hot reload
cd frontend
npm run dev
```

### **Testing**
```bash
# Test API endpoints
curl http://localhost:8000/api/stats
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "session_id": "test123"}'
```

### **Database Management**
```bash
# Initialize database
python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"

# Check database status
python -c "from backend.db import check_connection; import asyncio; asyncio.run(check_connection())"
```

## 🔒 Security Considerations

- **API Keys**: Store securely in `.env` file
- **CORS**: Configure allowed origins for production
- **Rate Limiting**: Implement request throttling
- **Input Validation**: All inputs validated with Pydantic
- **SQL Injection**: Protected with SQLAlchemy ORM

## 📈 Performance Optimization

- **Async Operations**: Full async/await implementation
- **Connection Pooling**: Database connection optimization
- **Caching**: Implement Redis for frequently accessed data
- **CDN**: Use CDN for static assets in production
- **Compression**: Enable gzip compression

## 🐛 Troubleshooting

### **Common Issues**

1. **Database Connection Errors**
   ```bash
   # Reinitialize database
   python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"
   ```

2. **API Import Errors**
   ```bash
   # Install missing dependencies
   pip install gnews transformers torch
   ```

3. **Frontend Build Issues**
   ```bash
   # Clear cache and reinstall
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **CORS Issues**
   - Check `ALLOWED_ORIGINS` in `.env`
   - Verify frontend URL matches backend configuration

### **Logs and Debugging**
- Backend logs: Check terminal output
- Frontend logs: Browser developer console
- Database logs: SQLAlchemy echo mode

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Reddit API for data access
- Google Gemini for AI capabilities
- Hugging Face for pre-trained models
- FastAPI and React communities

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation at `/docs`

---

**Built with ❤️ for intelligent news analysis**
