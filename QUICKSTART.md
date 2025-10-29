# 🚀 AI News Assistant - Quick Start Guide

## ⚡ Get Started in 5 Minutes

### **Step 1: Prerequisites**
Ensure you have installed:
- Python 3.8+
- Node.js 16+
- Git

### **Step 2: Setup Environment**
```bash
# Clone the repository
git clone <your-repo-url>
cd Project-X

# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env
```

**Required API Keys:**
- Reddit API credentials
- Google Gemini API key

### **Step 3: Automated Setup**
```bash
# Run the setup script
python setup.py
```

This will:
- ✅ Check system requirements
- ✅ Install all dependencies
- ✅ Initialize the database
- ✅ Validate configuration

### **Step 4: Launch the System**
```bash
# Start both frontend and backend
python launcher.py
```

### **Step 5: Access the Application**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 Quick Test

1. **Open Dashboard**: Navigate to http://localhost:3000
2. **Test Chat**: Go to Chat tab and ask "What's trending in AI?"
3. **Sentiment Analysis**: Try analyzing "climate change" sentiment

## 🔧 Manual Setup (Alternative)

If automated setup fails:

```bash
# Backend setup
pip install -r requirements.txt
python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"

# Frontend setup
cd frontend
npm install

# Launch
cd ..
python launcher.py
```

## 🐛 Troubleshooting

### **Common Issues:**

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database errors**
   ```bash
   python -c "from backend.db import init_db; import asyncio; asyncio.run(init_db())"
   ```

3. **Frontend won't start**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **API connection issues**
   - Check if .env file exists and has correct API keys
   - Verify ports 3000 and 8000 are available

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [WORKFLOW.md](WORKFLOW.md) for system architecture
- Review [STRUCTURE.md](STRUCTURE.md) for file organization

## 🆘 Need Help?

- Check the troubleshooting section in README.md
- Review API documentation at http://localhost:8000/docs
- Create an issue in the repository

---

**🎉 You're all set! Start exploring the AI News Assistant!**
