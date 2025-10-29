import { motion } from 'framer-motion'

const MetricCard = ({ title, value, subtitle, trend, icon }) => {
  const getTrendColor = (trend) => {
    switch (trend) {
      case 'up': return 'text-green-400'
      case 'down': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return '↗'
      case 'down': return '↘'
      default: return '→'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-dark-300 mb-1">{title}</h3>
          <p className="text-2xl font-bold text-white">{value}</p>
          {subtitle && (
            <p className="text-xs text-dark-400 mt-1">{subtitle}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {icon && <div className="text-primary-400">{icon}</div>}
          {trend && (
            <span className={`text-sm ${getTrendColor(trend)}`}>
              {getTrendIcon(trend)}
            </span>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default MetricCard
