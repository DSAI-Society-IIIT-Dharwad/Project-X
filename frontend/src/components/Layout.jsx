import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  MessageCircle, 
  TrendingUp, 
  RefreshCw,
  Menu,
  X 
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import { refreshData } from '../services/api'
import useStore from '../store/useStore'

const Layout = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const [refreshing, setRefreshing] = useState(false)
  const { sidebarOpen, toggleSidebar } = useStore()

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/chat', icon: MessageCircle, label: 'Chat Assistant' },
    { path: '/sentiment', icon: TrendingUp, label: 'Sentiment Analysis' },
  ]

  const handleRefresh = async () => {
    setRefreshing(true)
    const toastId = toast.loading('Refreshing data...')
    
    try {
      const response = await refreshData()
      const data = response.data
      toast.success(
        `✅ Refresh complete!\nReddit: ${data.reddit.posts_saved} posts\nGoogle News: ${data.google_news.articles_saved} articles`,
        { id: toastId, duration: 5000 }
      )
    } catch (error) {
      toast.error('Failed to refresh data', { id: toastId })
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="w-72 bg-dark-800/95 backdrop-blur-xl border-r border-primary-500/20 flex flex-col"
          >
            {/* Logo */}
            <div className="p-6 border-b border-primary-500/20">
              <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                🤖 AI News
              </h1>
              <p className="text-sm text-dark-300 mt-1">
                Intelligent News Assistant
              </p>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.path
                
                return (
                  <button
                    key={item.path}
                    onClick={() => navigate(item.path)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-semibold transition-all duration-300 ${
                      isActive
                        ? 'bg-gradient-primary text-white shadow-lg shadow-primary-500/30'
                        : 'bg-dark-700/50 text-dark-200 hover:bg-dark-700 hover:text-dark-50'
                    }`}
                  >
                    <Icon size={20} />
                    <span>{item.label}</span>
                  </button>
                )
              })}
            </nav>

            {/* Actions */}
            <div className="p-4 border-t border-primary-500/20 space-y-4">
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                <RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />
                {refreshing ? 'Refreshing...' : 'Refresh Data'}
              </button>

              <div className="text-xs text-dark-300 leading-relaxed">
                <p>AI-powered assistant that analyzes Reddit and Google News.</p>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="bg-dark-800/80 backdrop-blur-xl border-b border-primary-500/20 px-6 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-lg hover:bg-dark-700 transition-colors"
            >
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>

            <div className="flex items-center gap-4">
              <div className="text-sm text-dark-300">
                {new Date().toLocaleDateString('en-US', { 
                  weekday: 'long', 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  )
}

export default Layout