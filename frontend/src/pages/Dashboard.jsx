import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { 
  PieChart, Pie, Cell, ResponsiveContainer, 
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend 
} from 'recharts'
import { getStats, getSentimentDistribution, getRecentPosts } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'
import PostCard from '../components/PostCard'
import PredictiveSentimentChart from '../components/PredictiveSentimentChart'
import TrendingNewsSummary from '../components/TrendingNewsSummary'

const COLORS = {
  positive: '#28a745',
  neutral: '#ffc107',
  negative: '#dc3545',
}

const Dashboard = () => {
  const [stats, setStats] = useState(null)
  const [redditSentiment, setRedditSentiment] = useState(null)
  const [newsSentiment, setNewsSentiment] = useState(null)
  const [redditPosts, setRedditPosts] = useState([])
  const [newsPosts, setNewsPosts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [statsRes, redditSentRes, newsSentRes, redditPostsRes, newsPostsRes] = 
        await Promise.all([
          getStats(),
          getSentimentDistribution('reddit'),
          getSentimentDistribution('google_news'),
          getRecentPosts('reddit', 5),
          getRecentPosts('google_news', 5),
        ])

      setStats(statsRes.data)
      setRedditSentiment(redditSentRes.data)
      setNewsSentiment(newsSentRes.data)
      setRedditPosts(redditPostsRes.data)
      setNewsPosts(newsPostsRes.data)
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <LoadingSpinner />

  const getSentimentIcon = (score) => {
    if (score > 0.2) return <TrendingUp className="text-green-400" />
    if (score < -0.2) return <TrendingDown className="text-red-400" />
    return <Minus className="text-yellow-400" />
  }

  const getSentimentEmoji = (score) => {
    if (score > 0.2) return '😊'
    if (score < -0.2) return '😟'
    return '😐'
  }

  const sentimentDiff = Math.abs(
    stats.reddit.avg_sentiment - stats.google_news.avg_sentiment
  )

  const redditChartData = [
    { name: 'Positive', value: redditSentiment.positive, color: COLORS.positive },
    { name: 'Neutral', value: redditSentiment.neutral, color: COLORS.neutral },
    { name: 'Negative', value: redditSentiment.negative, color: COLORS.negative },
  ]

  const newsChartData = [
    { name: 'Positive', value: newsSentiment.positive, color: COLORS.positive },
    { name: 'Neutral', value: newsSentiment.neutral, color: COLORS.neutral },
    { name: 'Negative', value: newsSentiment.negative, color: COLORS.negative },
  ]

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-2"
      >
        <h1 className="text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent">
          Unified News Dashboard
        </h1>
        <p className="text-dark-300 text-lg">
          Compare Reddit discussions with Google News coverage
        </p>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="📱 Reddit Posts"
          value={stats.reddit.total.toLocaleString()}
          subtitle={`Last 24h: ${stats.reddit.recent_24h}`}
          trend="up"
        />
        <MetricCard
          title="📰 Google News"
          value={stats.google_news.total.toLocaleString()}
          subtitle={`Last 24h: ${stats.google_news.recent_24h}`}
          trend="up"
        />
        <MetricCard
          title={`${getSentimentEmoji(stats.reddit.avg_sentiment)} Reddit Sentiment`}
          value={stats.reddit.avg_sentiment.toFixed(3)}
          subtitle="Average score"
          icon={getSentimentIcon(stats.reddit.avg_sentiment)}
        />
        <MetricCard
          title={`${getSentimentEmoji(stats.google_news.avg_sentiment)} News Sentiment`}
          value={stats.google_news.avg_sentiment.toFixed(3)}
          subtitle="Average score"
          icon={getSentimentIcon(stats.google_news.avg_sentiment)}
        />
      </div>

      {/* Sentiment Comparison */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4">📊 Sentiment Comparison</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
          {/* Reddit Pie */}
          <div>
            <h3 className="text-lg font-semibold mb-4 text-center">Reddit Sentiment</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={redditChartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {redditChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    background: '#1a1f3a', 
                    border: '1px solid rgba(102, 126, 234, 0.3)',
                    borderRadius: '8px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* News Pie */}
          <div>
            <h3 className="text-lg font-semibold mb-4 text-center">Google News Sentiment</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={newsChartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {newsChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    background: '#1a1f3a', 
                    border: '1px solid rgba(102, 126, 234, 0.3)',
                    borderRadius: '8px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Sentiment Analysis */}
        <div className={`p-4 rounded-xl border ${
          sentimentDiff < 0.2 
            ? 'bg-green-500/10 border-green-500/30' 
            : 'bg-yellow-500/10 border-yellow-500/30'
        }`}>
          {sentimentDiff < 0.2 ? (
            <p className="text-green-400 font-semibold">
              ✅ Sentiments ALIGN - Reddit discussions match mainstream news coverage
            </p>
          ) : (
            <p className="text-yellow-400 font-semibold">
              ⚠️ Sentiments DIFFER - {sentimentDiff.toFixed(2)} difference between sources
            </p>
          )}
        </div>
      </motion.div>

      {/* Predictive Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <PredictiveSentimentChart />
      </motion.div>

      {/* Trending News Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <TrendingNewsSummary />
      </motion.div>

      {/* Recent Posts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reddit Posts */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <h2 className="text-2xl font-bold mb-4">📱 Recent Reddit Posts</h2>
          <div className="space-y-3">
            {redditPosts.map((post) => (
              <PostCard key={post.id} post={post} />
            ))}
          </div>
        </motion.div>

        {/* News Posts */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <h2 className="text-2xl font-bold mb-4">📰 Recent Google News</h2>
          <div className="space-y-3">
            {newsPosts.map((post) => (
              <PostCard key={post.id} post={post} isNews />
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Dashboard