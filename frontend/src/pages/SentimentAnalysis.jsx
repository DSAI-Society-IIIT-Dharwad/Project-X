import { useState } from 'react'
import { motion } from 'framer-motion'
import { Search, TrendingUp } from 'lucide-react'
import { analyzeSentiment } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import SentimentResults from '../components/SentimentResults'
import toast from 'react-hot-toast'

const SentimentAnalysis = () => {
  const [topic, setTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)

  const exampleTopics = [
    'artificial intelligence',
    'climate change',
    'cryptocurrency',
    'electric vehicles',
    'remote work'
  ]

  const handleAnalyze = async () => {
    if (!topic.trim() || loading) return

    setLoading(true)
    setResults(null)
    const toastId = toast.loading('Analyzing sentiment...')

    try {
      const response = await analyzeSentiment(topic)
      const data = response.data

      if (data.success) {
        setResults(data)
        toast.success('Analysis complete!', { id: toastId })
      } else {
        toast.error('No posts found for this topic', { id: toastId })
      }
    } catch (error) {
      toast.error('Analysis failed: ' + error.message, { id: toastId })
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAnalyze()
    }
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-2"
      >
        <h1 className="text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent">
          📈 Sentiment Analysis
        </h1>
        <p className="text-dark-300 text-lg">
          Analyze public opinion across Reddit
        </p>
      </motion.div>

      {/* Search Bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search 
              className="absolute left-4 top-1/2 transform -translate-y-1/2 text-dark-400" 
              size={20} 
            />
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter topic to analyze (e.g., climate change, AI, cryptocurrency)"
              disabled={loading}
              className="input-field pl-12"
            />
          </div>
          <button
            onClick={handleAnalyze}
            disabled={loading || !topic.trim()}
            className="btn-primary px-8 flex items-center gap-2"
          >
            <TrendingUp size={18} />
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {/* Example Topics */}
        <div className="mt-4">
          <p className="text-sm text-dark-400 mb-2">Try these examples:</p>
          <div className="flex flex-wrap gap-2">
            {exampleTopics.map((example) => (
              <button
                key={example}
                onClick={() => setTopic(example)}
                disabled={loading}
                className="px-3 py-1.5 rounded-lg bg-dark-600 hover:bg-dark-500 transition-colors text-sm"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Loading */}
      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <LoadingSpinner message="Searching Reddit and analyzing sentiment..." />
        </motion.div>
      )}

      {/* Results */}
      {results && !loading && (
        <SentimentResults results={results} topic={topic} />
      )}

      {/* Empty State */}
      {!results && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card text-center py-16"
        >
          <div className="text-6xl mb-4">📊</div>
          <h2 className="text-2xl font-bold mb-2">Ready to Analyze</h2>
          <p className="text-dark-300 max-w-md mx-auto">
            Enter any topic above to see detailed sentiment analysis from Reddit discussions.
            We'll analyze posts, comments, and engagement to give you comprehensive insights.
          </p>
        </motion.div>
      )}
    </div>
  )
}

export default SentimentAnalysis