import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Dashboard APIs
export const getStats = () => api.get('/api/stats')

export const getSentimentDistribution = (source) => 
  api.get(`/api/sentiment-distribution/${source}`)

export const getRecentPosts = (source, limit = 10) => 
  api.get(`/api/recent-posts/${source}?limit=${limit}`)

export const refreshData = () => api.post('/api/refresh')

export const compareSentiment = (topic) => 
  api.post('/api/compare-sentiment', { topic })

// Chat APIs
export const sendChatMessage = (query, sessionId) => 
  api.post('/api/chat', { query, session_id: sessionId })

export const getChatHistory = (sessionId, limit = 10) => 
  api.get(`/api/chat-history/${sessionId}?limit=${limit}`)

// Sentiment Analysis APIs
export const analyzeSentiment = (topic) => 
  api.post('/api/analyze-sentiment', { topic })

// Predictive Analysis APIs
export const getPredictiveSentiment = () => 
  api.get('/api/predictive-sentiment')

// Trending News APIs
export const getTrendingSummary = () => 
  api.get('/api/trending-summary')

export default api