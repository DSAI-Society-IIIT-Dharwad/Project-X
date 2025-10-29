import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { getPredictiveSentiment } from '../services/api'
import LoadingSpinner from './LoadingSpinner'

const PredictiveSentimentChart = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchPredictiveData()
  }, [])

  const fetchPredictiveData = async () => {
    try {
      setLoading(true)
      const response = await getPredictiveSentiment()
      setData(response.data)
    } catch (err) {
      setError('Failed to fetch predictive data')
      console.error('Predictive data error:', err)
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
            onClick={fetchPredictiveData}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!data) return null

  // Prepare chart data
  const chartData = []
  
  // Combine historical and predicted data
  const redditHistorical = data.reddit.historical || []
  const redditPredictions = data.reddit.predictions || []
  const gnewsHistorical = data.google_news.historical || []
  const gnewsPredictions = data.google_news.predictions || []

  // Create combined dataset
  const allDates = new Set([
    ...redditHistorical.map(d => d.date),
    ...redditPredictions.map(d => d.date),
    ...gnewsHistorical.map(d => d.date),
    ...gnewsPredictions.map(d => d.date)
  ])

  Array.from(allDates).sort().forEach(date => {
    const redditHist = redditHistorical.find(d => d.date === date)
    const redditPred = redditPredictions.find(d => d.date === date)
    const gnewsHist = gnewsHistorical.find(d => d.date === date)
    const gnewsPred = gnewsPredictions.find(d => d.date === date)

    chartData.push({
      date: new Date(date).toLocaleDateString(),
      redditHistorical: redditHist?.sentiment || null,
      redditPredicted: redditPred?.predicted_sentiment || null,
      gnewsHistorical: gnewsHist?.sentiment || null,
      gnewsPredicted: gnewsPred?.predicted_sentiment || null,
      redditConfidence: redditPred?.confidence || null,
      gnewsConfidence: gnewsPred?.confidence || null
    })
  })

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.dataKey.includes('Historical') ? '📊' : '🔮'} {entry.dataKey.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}: {entry.value?.toFixed(3) || 'N/A'}
              {entry.dataKey.includes('Predicted') && entry.payload[`${entry.dataKey.replace('Predicted', '')}Confidence`] && (
                <span className="text-gray-500 ml-2">
                  (Confidence: {(entry.payload[`${entry.dataKey.replace('Predicted', '')}Confidence`] * 100).toFixed(0)}%)
                </span>
              )}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">🔮 Predictive Sentiment Analysis</h2>
        <button 
          onClick={fetchPredictiveData}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Refresh
        </button>
      </div>

      <div className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">Reddit Trend</h3>
            <div className="flex items-center">
              <span className={`text-2xl mr-2 ${data.reddit.trend === 'increasing' ? '📈' : '📉'}`}>
                {data.reddit.trend === 'increasing' ? '📈' : '📉'}
              </span>
              <span className={`font-bold ${data.reddit.trend === 'increasing' ? 'text-green-600' : 'text-red-600'}`}>
                {data.reddit.trend === 'increasing' ? 'Increasing' : 'Decreasing'}
              </span>
              <span className="ml-2 text-sm text-gray-600">
                (Strength: {data.analysis.reddit_trend_strength.toFixed(3)})
              </span>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 mb-2">Google News Trend</h3>
            <div className="flex items-center">
              <span className={`text-2xl mr-2 ${data.google_news.trend === 'increasing' ? '📈' : '📉'}`}>
                {data.google_news.trend === 'increasing' ? '📈' : '📉'}
              </span>
              <span className={`font-bold ${data.google_news.trend === 'increasing' ? 'text-green-600' : 'text-red-600'}`}>
                {data.google_news.trend === 'increasing' ? 'Increasing' : 'Decreasing'}
              </span>
              <span className="ml-2 text-sm text-gray-600">
                (Strength: {data.analysis.gnews_trend_strength.toFixed(3)})
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis 
              domain={[-1, 1]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Sentiment Score', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {/* Reddit Historical */}
            <Area
              type="monotone"
              dataKey="redditHistorical"
              stackId="1"
              stroke="#3B82F6"
              fill="#3B82F6"
              fillOpacity={0.3}
              name="Reddit Historical"
              strokeWidth={2}
            />
            
            {/* Reddit Predicted */}
            <Area
              type="monotone"
              dataKey="redditPredicted"
              stackId="2"
              stroke="#1D4ED8"
              fill="#1D4ED8"
              fillOpacity={0.2}
              strokeDasharray="5 5"
              name="Reddit Predicted"
              strokeWidth={2}
            />
            
            {/* Google News Historical */}
            <Area
              type="monotone"
              dataKey="gnewsHistorical"
              stackId="3"
              stroke="#10B981"
              fill="#10B981"
              fillOpacity={0.3}
              name="Google News Historical"
              strokeWidth={2}
            />
            
            {/* Google News Predicted */}
            <Area
              type="monotone"
              dataKey="gnewsPredicted"
              stackId="4"
              stroke="#059669"
              fill="#059669"
              fillOpacity={0.2}
              strokeDasharray="5 5"
              name="Google News Predicted"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>📊 <strong>Historical:</strong> Past sentiment data from your API</p>
        <p>🔮 <strong>Predicted:</strong> AI-powered 7-day sentiment forecast</p>
        <p>📈 <strong>Trend:</strong> Overall direction based on linear regression analysis</p>
      </div>
    </div>
  )
}

export default PredictiveSentimentChart
