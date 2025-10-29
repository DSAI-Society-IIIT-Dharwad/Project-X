import React, { useState, useEffect } from 'react'
import { getTrendingSummary } from '../services/api'
import LoadingSpinner from './LoadingSpinner'
import PostCard from './PostCard'

const TrendingNewsSummary = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('summary')

  useEffect(() => {
    fetchTrendingData()
  }, [])

  const fetchTrendingData = async () => {
    try {
      setLoading(true)
      const response = await getTrendingSummary()
      setData(response.data)
    } catch (err) {
      setError('Failed to fetch trending data')
      console.error('Trending data error:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center h-64">
          <LoadingSpinner />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center text-red-600">
          <p>{error}</p>
          <button 
            onClick={fetchTrendingData}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!data) return null

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-100'
      case 'negative': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive': return '😊'
      case 'negative': return '😞'
      default: return '😐'
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">📈 Trending News Summary</h2>
        <button 
          onClick={fetchTrendingData}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Sentiment Overview */}
      <div className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 mb-2">Overall Sentiment</h3>
            <div className="flex items-center">
              <span className="text-2xl mr-2">
                {getSentimentIcon(data.analysis.overall_sentiment)}
              </span>
              <span className={`font-bold text-lg ${getSentimentColor(data.analysis.overall_sentiment).split(' ')[0]}`}>
                {data.analysis.overall_sentiment.charAt(0).toUpperCase() + data.analysis.overall_sentiment.slice(1)}
              </span>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">Reddit Posts</h3>
            <div className="text-2xl font-bold text-blue-600">
              {data.analysis.total_reddit_posts}
            </div>
            <div className="text-sm text-gray-600">Last 24 hours</div>
          </div>
          
          <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-800 mb-2">Google News</h3>
            <div className="text-2xl font-bold text-purple-600">
              {data.analysis.total_gnews_posts}
            </div>
            <div className="text-sm text-gray-600">Last 24 hours</div>
          </div>
        </div>

        {/* Sentiment Distribution */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-800 mb-3">Reddit Sentiment Distribution</h4>
            <div className="space-y-2">
              {Object.entries(data.sentiment_distribution.reddit).map(([sentiment, count]) => (
                <div key={sentiment} className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="mr-2">{getSentimentIcon(sentiment)}</span>
                    <span className="capitalize">{sentiment}</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-20 bg-gray-200 rounded-full h-2 mr-2">
                      <div 
                        className={`h-2 rounded-full ${sentiment === 'positive' ? 'bg-green-500' : sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'}`}
                        style={{ width: `${(count / data.analysis.total_reddit_posts) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-800 mb-3">Google News Sentiment Distribution</h4>
            <div className="space-y-2">
              {Object.entries(data.sentiment_distribution.google_news).map(([sentiment, count]) => (
                <div key={sentiment} className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="mr-2">{getSentimentIcon(sentiment)}</span>
                    <span className="capitalize">{sentiment}</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-20 bg-gray-200 rounded-full h-2 mr-2">
                      <div 
                        className={`h-2 rounded-full ${sentiment === 'positive' ? 'bg-green-500' : sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'}`}
                        style={{ width: `${(count / data.analysis.total_gnews_posts) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('summary')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'summary'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              📝 AI Summary
            </button>
            <button
              onClick={() => setActiveTab('reddit')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'reddit'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              🔴 Reddit Trending
            </button>
            <button
              onClick={() => setActiveTab('gnews')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'gnews'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              📰 Google News Trending
            </button>
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-64">
        {activeTab === 'summary' && (
          <div className="prose max-w-none">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border-l-4 border-blue-500">
              <h3 className="text-lg font-semibold text-blue-800 mb-3">🤖 AI-Generated Summary</h3>
              <div className="text-gray-700 whitespace-pre-line leading-relaxed">
                {data.summary}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'reddit' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Top Reddit Posts (Last 24h)</h3>
            {data.reddit_trending.map((post, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-gray-800 flex-1 mr-4">{post.title}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(post.sentiment)}`}>
                    {getSentimentIcon(post.sentiment)} {post.sentiment}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <div className="flex items-center space-x-4">
                    <span>r/{post.subreddit}</span>
                    <span>👍 {post.score}</span>
                    <span>💬 {post.comments}</span>
                    <span>Score: {post.sentiment_score.toFixed(2)}</span>
                  </div>
                  <a 
                    href={post.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800"
                  >
                    View Post →
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'gnews' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Top Google News Articles (Last 24h)</h3>
            {data.gnews_trending.map((article, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-gray-800 flex-1 mr-4">{article.title}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(article.sentiment)}`}>
                    {getSentimentIcon(article.sentiment)} {article.sentiment}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <div className="flex items-center space-x-4">
                    <span>Score: {article.sentiment_score.toFixed(2)}</span>
                    <span>{new Date(article.created_at).toLocaleString()}</span>
                  </div>
                  <a 
                    href={article.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800"
                  >
                    Read Article →
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default TrendingNewsSummary
