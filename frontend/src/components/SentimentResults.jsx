import { motion } from 'framer-motion'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
} from 'recharts'
import PostCard from './PostCard'

const COLORS = {
  positive: '#28a745',
  neutral: '#ffc107',
  negative: '#dc3545',
}

const SentimentResults = ({ results, topic }) => {
  const { analysis, sample_posts } = results

  const chartData = [
    { name: 'Positive', value: analysis.sentiment.positive, color: COLORS.positive },
    { name: 'Neutral', value: analysis.sentiment.neutral, color: COLORS.neutral },
    { name: 'Negative', value: analysis.sentiment.negative, color: COLORS.negative },
  ]

  const subredditData = Object.entries(analysis.subreddits || {})
    .slice(0, 10)
    .map(([name, count]) => ({ name: `r/${name}`, count }))

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-2">
          Analysis for "{topic}"
        </h2>
        <p className="text-dark-300">
          Based on {analysis.total_posts} posts from Reddit
        </p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <p className="text-sm text-dark-300 mb-2">Total Posts</p>
          <p className="text-3xl font-bold text-primary-400">{analysis.total_posts}</p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-dark-300 mb-2">😊 Positive</p>
          <p className="text-3xl font-bold text-green-400">
            {((analysis.sentiment.positive / analysis.total_posts) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-dark-300 mb-2">😟 Negative</p>
          <p className="text-3xl font-bold text-red-400">
            {((analysis.sentiment.negative / analysis.total_posts) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-dark-300 mb-2">😐 Neutral</p>
          <p className="text-3xl font-bold text-yellow-400">
            {((analysis.sentiment.neutral / analysis.total_posts) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Sentiment Score */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4">Overall Sentiment Score</h3>
        <div className="relative">
          <div className="h-8 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"></div>
          <div
            className="absolute top-0 w-1 h-10 bg-white shadow-lg -mt-1"
            style={{ left: `${((analysis.sentiment.average_score + 1) / 2) * 100}%` }}
          ></div>
        </div>
        <div className="flex justify-between text-xs text-dark-400 mt-2">
          <span>-1.0 (Very Negative)</span>
          <span className="text-2xl font-bold text-primary-400">
            {analysis.sentiment.average_score.toFixed(3)}
          </span>
          <span>+1.0 (Very Positive)</span>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sentiment Distribution */}
        <div className="card">
          <h3 className="text-xl font-bold mb-4">Sentiment Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                dataKey="value"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: '#1a1f3a',
                  border: '1px solid rgba(102, 126, 234, 0.3)',
                  borderRadius: '8px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top Subreddits */}
        <div className="card">
          <h3 className="text-xl font-bold mb-4">Top Subreddits</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={subredditData} layout="vertical">
              <XAxis type="number" stroke="#8b92ab" />
              <YAxis dataKey="name" type="category" width={100} stroke="#8b92ab" />
              <Tooltip
                contentStyle={{
                  background: '#1a1f3a',
                  border: '1px solid rgba(102, 126, 234, 0.3)',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="count" fill="#667eea" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Keywords */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4">🔑 Top Keywords</h3>
        <div className="flex flex-wrap gap-2">
          {analysis.keywords?.slice(0, 20).map((keyword, idx) => (
            <span
              key={idx}
              className="px-4 py-2 rounded-full bg-gradient-primary text-white font-semibold text-sm"
            >
              {keyword}
            </span>
          ))}
        </div>
      </div>

      {/* Sample Posts */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4">📋 Sample Posts</h3>
        <div className="space-y-3">
          {sample_posts?.slice(0, 10).map((post, idx) => (
            <PostCard key={idx} post={post} />
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default SentimentResults
