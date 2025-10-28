import React, { useState, useEffect, createContext, useContext } from 'react';
import { BarChart3, TrendingUp, MessageSquare, Info, Moon, Sun, Home, Activity, Send, Loader2, AlertCircle } from 'lucide-react';
import './App.css';

// Theme Context
const ThemeContext = createContext();
const useTheme = () => useContext(ThemeContext);

// Chat Context
const ChatContext = createContext();
const useChat = () => useContext(ChatContext);

// API Configuration - UPDATE THIS TO YOUR BACKEND URL
const API_BASE = 'http://localhost:8000';

const api = {
  getDashboard: async () => {
    const res = await fetch(`${API_BASE}/stats/dashboard`);
    if (!res.ok) throw new Error('Failed to fetch dashboard data');
    return res.json();
  },
  getTrends: async () => {
    const res = await fetch(`${API_BASE}/trending`);
    if (!res.ok) throw new Error('Failed to fetch trends');
    return res.json();
  },
  getSentiment: async () => {
    const res = await fetch(`${API_BASE}/sentiment/stats`);
    if (!res.ok) throw new Error('Failed to fetch sentiment');
    return res.json();
  },
  sendMessage: async (msg) => {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });
    if (!res.ok) throw new Error('Failed to send message');
    return res.json();
  }
};

// Theme Provider
function ThemeProvider({ children }) {
  const [dark, setDark] = useState(true);
  return (
    <ThemeContext.Provider value={{ dark, setDark }}>
      {children}
    </ThemeContext.Provider>
  );
}

// Chat Provider
function ChatProvider({ children }) {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! I can help you analyze trends and sentiment. Ask me anything!', time: new Date() }
  ]);
  
  const addMessage = (msg) => setMessages(prev => [...prev, msg]);
  
  return (
    <ChatContext.Provider value={{ messages, addMessage }}>
      {children}
    </ChatContext.Provider>
  );
}

// Navbar Component
function Navbar() {
  const { dark, setDark } = useTheme();
  const [isLive, setIsLive] = useState(false);
  
  useEffect(() => {
    // Check backend connection
    fetch(`${API_BASE}/stats/dashboard`)
      .then(() => setIsLive(true))
      .catch(() => setIsLive(false));
  }, []);
  
  return (
    <nav className="bg-gray-900 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
          <TrendingUp className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-white">TrendSense AI</h1>
          <p className="text-xs text-gray-400">Real-time Analytics Dashboard</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className={`flex items-center gap-2 px-3 py-1 ${isLive ? 'bg-green-500/10 border-green-500/20' : 'bg-red-500/10 border-red-500/20'} border rounded-full`}>
          <div className={`w-2 h-2 ${isLive ? 'bg-green-500' : 'bg-red-500'} rounded-full ${isLive ? 'animate-pulse' : ''}`} />
          <span className={`text-xs ${isLive ? 'text-green-400' : 'text-red-400'}`}>
            {isLive ? 'Live' : 'Offline'}
          </span>
        </div>
        <button
          onClick={() => setDark(!dark)}
          className="p-2 hover:bg-gray-800 rounded-lg transition"
        >
          {dark ? <Sun className="w-5 h-5 text-gray-400" /> : <Moon className="w-5 h-5 text-gray-400" />}
        </button>
      </div>
    </nav>
  );
}

// Sidebar Component
function Sidebar({ activePage, setActivePage }) {
  const pages = [
    { id: 'dashboard', icon: Home, label: 'Dashboard' },
    { id: 'chatbot', icon: MessageSquare, label: 'AI Chat' },
    { id: 'about', icon: Info, label: 'About' }
  ];
  
  return (
    <aside className="w-64 bg-gray-900 border-r border-gray-800 p-4">
      <div className="space-y-2">
        {pages.map(page => (
          <button
            key={page.id}
            onClick={() => setActivePage(page.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition ${
              activePage === page.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <page.icon className="w-5 h-5" />
            <span className="font-medium">{page.label}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}

// Error Display Component
function ErrorDisplay({ message, retry }) {
  return (
    <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-6 flex items-start gap-4">
      <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
      <div className="flex-1">
        <h3 className="text-red-400 font-semibold mb-2">Error Loading Data</h3>
        <p className="text-gray-400 text-sm mb-4">{message}</p>
        {retry && (
          <button
            onClick={retry}
            className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm transition"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

// Dashboard Cards
function StatCard({ title, value, change, icon: Icon, loading }) {
  if (loading) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 animate-pulse">
        <div className="h-20"></div>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 hover:border-blue-500/50 transition">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-400 text-sm mb-2">{title}</p>
          <h3 className="text-3xl font-bold text-white mb-1">{value}</h3>
          {change !== undefined && (
            <p className={`text-sm ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {change >= 0 ? '↑' : '↓'} {Math.abs(change)}% from last period
            </p>
          )}
        </div>
        <div className="p-3 bg-blue-500/10 rounded-lg">
          <Icon className="w-6 h-6 text-blue-400" />
        </div>
      </div>
    </div>
  );
}

// Dashboard Page
function Dashboard() {
  const [dashboardData, setDashboardData] = useState(null);
  const [trendsData, setTrendsData] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [dashboard, trends, sentiment] = await Promise.all([
        api.getDashboard(),
        api.getTrends(),
        api.getSentiment()
      ]);
      setDashboardData(dashboard);
      setTrendsData(trends);
      setSentimentData(sentiment);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);
  
  if (error) {
    return <ErrorDisplay message={error} retry={fetchData} />;
  }
  
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Dashboard Overview</h2>
        <p className="text-gray-400">Real-time insights and analytics</p>
      </div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard 
          title="Total Posts" 
          value={dashboardData?.total_posts || 0}
          change={dashboardData?.posts_change}
          icon={BarChart3}
          loading={loading}
        />
        <StatCard 
          title="Active Topics" 
          value={dashboardData?.active_topics || 0}
          change={dashboardData?.topics_change}
          icon={TrendingUp}
          loading={loading}
        />
        <StatCard 
          title="Positive Sentiment" 
          value={sentimentData?.positive_percentage ? `${sentimentData.positive_percentage}%` : '0%'}
          change={sentimentData?.sentiment_change}
          icon={Activity}
          loading={loading}
        />
      </div>
      
      {/* Sentiment Distribution */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Sentiment Distribution</h3>
        {loading ? (
          <div className="animate-pulse space-y-3">
            <div className="h-12 bg-gray-700 rounded"></div>
            <div className="h-12 bg-gray-700 rounded"></div>
            <div className="h-12 bg-gray-700 rounded"></div>
          </div>
        ) : sentimentData ? (
          <div className="space-y-3">
            {[
              { 
                label: 'Positive', 
                value: sentimentData.positive_percentage || 0, 
                color: 'bg-green-500',
                count: sentimentData.positive_count || 0
              },
              { 
                label: 'Neutral', 
                value: sentimentData.neutral_percentage || 0, 
                color: 'bg-gray-500',
                count: sentimentData.neutral_count || 0
              },
              { 
                label: 'Negative', 
                value: sentimentData.negative_percentage || 0, 
                color: 'bg-red-500',
                count: sentimentData.negative_count || 0
              }
            ].map(item => (
              <div key={item.label}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">{item.label} ({item.count})</span>
                  <span className="text-white font-medium">{item.value.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div className={`${item.color} h-2 rounded-full transition-all duration-500`} style={{ width: `${item.value}%` }} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400">No sentiment data available</p>
        )}
      </div>
      
      {/* Trending Topics */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Trending Topics</h3>
        {loading ? (
          <div className="animate-pulse space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-20 bg-gray-700 rounded"></div>
            ))}
          </div>
        ) : trendsData && trendsData.length > 0 ? (
          <div className="space-y-3">
            {trendsData.slice(0, 5).map((trend, i) => (
              <div key={i} className="bg-gray-700/50 border border-gray-600 rounded-xl p-4 hover:border-blue-500/50 transition">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="text-lg font-semibold text-white mb-1">
                      {trend.topic || trend.name || 'Unknown Topic'}
                    </h4>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-gray-400">
                        {trend.count || trend.mentions || 0} mentions
                      </span>
                      {trend.sentiment !== undefined && (
                        <span className={trend.sentiment > 0 ? 'text-green-400' : trend.sentiment < 0 ? 'text-red-400' : 'text-gray-400'}>
                          Sentiment: {(trend.sentiment * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                  {trend.trend && (
                    <div className={`px-3 py-1 rounded-full text-sm ${
                      trend.trend === 'up' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                    }`}>
                      {trend.trend === 'up' ? '↑' : '↓'} {trend.trend}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400">No trending topics available</p>
        )}
      </div>
    </div>
  );
}

// Chatbot Page
function Chatbot() {
  const { messages, addMessage } = useChat();
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const send = async () => {
    if (!input.trim()) return;
    
    const userMessage = input;
    addMessage({ role: 'user', text: userMessage, time: new Date() });
    setInput('');
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.sendMessage(userMessage);
      addMessage({
        role: 'bot',
        text: response.answer || response.response || response.message || 'No response',
        sources: response.sources,
        time: new Date()
      });
    } catch (err) {
      setError(err.message);
      addMessage({
        role: 'bot',
        text: `Sorry, I encountered an error: ${err.message}`,
        time: new Date()
      });
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-12rem)]">
      <h2 className="text-2xl font-bold text-white mb-4">AI Assistant</h2>
      
      <div className="flex-1 bg-gray-800/50 border border-gray-700 rounded-xl p-4 overflow-y-auto mb-4">
        <div className="space-y-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] px-4 py-3 rounded-2xl ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-100'
              }`}>
                <p className="text-sm whitespace-pre-wrap">{m.text}</p>
                {m.sources && m.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-600">
                    <p className="text-xs text-gray-400 mb-1">Sources:</p>
                    {m.sources.map((src, j) => (
                      <p key={j} className="text-xs text-blue-300">• {src}</p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-700 px-4 py-3 rounded-2xl">
                <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
              </div>
            </div>
          )}
        </div>
      </div>
      
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
          Error: {error}
        </div>
      )}
      
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !loading && send()}
          placeholder="Ask about trends, sentiment, or insights..."
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          disabled={loading}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg transition flex items-center gap-2"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

// About Page
function About() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white">About TrendSense AI</h2>
      <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-8 text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <TrendingUp className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-2xl font-bold text-white mb-4">TrendSense AI Dashboard</h3>
        <p className="text-gray-400 text-lg mb-6 max-w-2xl mx-auto">
          Real-time trend and sentiment analysis platform powered by AI
        </p>
        <div className="flex justify-center gap-4 text-sm text-gray-400">
          <span>Version 1.0.0</span>
          <span>•</span>
          <span>Built with React & FastAPI</span>
        </div>
      </div>
    </div>
  );
}

// Main App
export default function App() {
  const [page, setPage] = useState('dashboard');
  
  const renderPage = () => {
    switch(page) {
      case 'dashboard': return <Dashboard />;
      case 'chatbot': return <Chatbot />;
      case 'about': return <About />;
      default: return <Dashboard />;
    }
  };
  
  return (
    <ThemeProvider>
      <ChatProvider>
        <div className="flex bg-gray-950 min-h-screen">
          <Sidebar activePage={page} setActivePage={setPage} />
          <div className="flex-1 flex flex-col">
            <Navbar />
            <main className="p-6 flex-1 overflow-y-auto">
              {renderPage()}
            </main>
          </div>
        </div>
      </ChatProvider>
    </ThemeProvider>
  );
}