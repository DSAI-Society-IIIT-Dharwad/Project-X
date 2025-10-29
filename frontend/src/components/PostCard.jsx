import { motion } from 'framer-motion'
import { ExternalLink } from 'lucide-react'

const PostCard = ({ post, isNews = false }) => {
  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'text-green-400 bg-green-400/10'
      case 'negative': return 'text-red-400 bg-red-400/10'
      default: return 'text-yellow-400 bg-yellow-400/10'
    }
  }

  const getSentimentEmoji = (sentiment) => {
    switch (sentiment) {
      case 'positive': return '😊'
      case 'negative': return '😟'
      default: return '😐'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-lg bg-dark-700/50 border border-primary-500/20 hover:border-primary-500/40 transition-colors"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-white mb-2 line-clamp-2">
            {post.title}
          </h3>
          
          <div className="flex items-center gap-3 text-sm text-dark-300 mb-2">
            <span className="font-medium">
              {isNews ? '📰 News' : `📱 r/${post.subreddit}`}
            </span>
            <span>•</span>
            <span>{post.score} points</span>
            {post.num_comments && (
              <>
                <span>•</span>
                <span>{post.num_comments} comments</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(post.sentiment)}`}>
              {getSentimentEmoji(post.sentiment)} {post.sentiment}
            </span>
            {post.sentiment_score && (
              <span className="text-xs text-dark-400">
                ({post.sentiment_score.toFixed(2)})
              </span>
            )}
          </div>
        </div>

        {post.url && (
          <a
            href={post.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 p-2 rounded-lg bg-dark-600 hover:bg-dark-500 transition-colors"
          >
            <ExternalLink size={16} className="text-primary-400" />
          </a>
        )}
      </div>
    </motion.div>
  )
}

export default PostCard
