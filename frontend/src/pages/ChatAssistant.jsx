import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Bot, User } from 'lucide-react'
import { sendChatMessage } from '../services/api'
import useStore from '../store/useStore'
import LoadingDots from '../components/LoadingDots'

const ChatAssistant = () => {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const { messages, addMessage, sessionId } = useStore()
  const [isLoading, setIsLoading] = useState(false)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = {
      type: 'user',
      text: input,
      timestamp: new Date(),
    }

    addMessage(userMessage)
    setInput('')
    setIsLoading(true)

    try {
      const response = await sendChatMessage(input, sessionId)
      const data = response.data

      addMessage({
        type: 'bot',
        text: data.answer,
        analysis: data.analysis,
        confidence: data.confidence,
        postsFound: data.posts_found,
        timestamp: new Date(),
      })
    } catch (error) {
      addMessage({
        type: 'bot',
        text: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="max-w-5xl mx-auto h-full flex flex-col">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-6"
      >
        <h1 className="text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-2">
          💬 AI Chat Assistant
        </h1>
        <p className="text-dark-300 text-lg">
          Ask anything! I'll search Reddit and provide intelligent answers
        </p>
      </motion.div>

      {/* Messages Container */}
      <div className="flex-1 card overflow-y-auto mb-4">
        <AnimatePresence initial={false}>
          {messages.length === 0 && !isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-full text-center"
            >
              <div className="text-6xl mb-4">👋</div>
              <h2 className="text-2xl font-bold mb-2">Welcome!</h2>
              <p className="text-dark-300 max-w-md">
                Hi! Ask me anything about current news and topics.
                I'll search Reddit in real-time and give you insights!
              </p>
              <div className="mt-6 grid grid-cols-2 gap-3">
                {[
                  "What's new with AI?",
                  "Climate change news",
                  "Cryptocurrency trends",
                  "Space exploration updates"
                ].map((example) => (
                  <button
                    key={example}
                    onClick={() => setInput(example)}
                    className="px-4 py-2 rounded-lg bg-dark-700 hover:bg-dark-600 transition-colors text-sm"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className={`mb-4 flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-3 max-w-[80%] ${message.type === 'user' ? 'flex-row-reverse' : ''}`}>
                {/* Avatar */}
                <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === 'user' 
                    ? 'bg-gradient-primary' 
                    : 'bg-dark-600 border-2 border-primary-500/30'
                }`}>
                  {message.type === 'user' ? <User size={20} /> : <Bot size={20} />}
                </div>

                {/* Message Content */}
                <div className="flex-1">
                  <div className={`p-4 rounded-2xl ${
                    message.type === 'user'
                      ? 'bg-gradient-primary text-white'
                      : 'bg-dark-700 border border-primary-500/20'
                  }`}>
                    <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
                  </div>

                  {/* Analysis for bot messages */}
                  {message.type === 'bot' && message.analysis && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-3 p-4 bg-dark-600/50 rounded-xl border border-primary-500/20"
                    >
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-primary-400">
                            {message.postsFound}
                          </div>
                          <div className="text-xs text-dark-300">Posts Found</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-primary-400">
                            {message.analysis.sentiment.average_score.toFixed(2)}
                          </div>
                          <div className="text-xs text-dark-300">Avg Sentiment</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-primary-400">
                            {(message.confidence * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-dark-300">Confidence</div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* Timestamp */}
                  <div className="text-xs text-dark-400 mt-2">
                    {message.timestamp?.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 flex justify-start"
            >
              <div className="flex gap-3">
                <div className="w-10 h-10 rounded-full bg-dark-600 border-2 border-primary-500/30 flex items-center justify-center">
                  <Bot size={20} />
                </div>
                <div className="p-4 rounded-2xl bg-dark-700 border border-primary-500/20">
                  <LoadingDots />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex gap-3"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me anything about current news..."
          disabled={isLoading}
          className="input-field flex-1"
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          className="btn-primary px-8 flex items-center gap-2"
        >
          <Send size={18} />
          Send
        </button>
      </motion.div>
    </div>
  )
}

export default ChatAssistant