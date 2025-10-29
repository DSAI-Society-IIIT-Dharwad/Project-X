import { create } from 'zustand'
import { v4 as uuidv4 } from 'uuid'

const useStore = create((set) => ({
  // Session
  sessionId: uuidv4(),
  
  // Dashboard state
  stats: null,
  setStats: (stats) => set({ stats }),
  
  // Chat state
  messages: [],
  addMessage: (message) => set((state) => ({ 
    messages: [...state.messages, message] 
  })),
  clearMessages: () => set({ messages: [] }),
  
  // Sentiment analysis state
  sentimentResults: null,
  setSentimentResults: (results) => set({ sentimentResults: results }),
  
  // Loading states
  loading: {
    dashboard: false,
    chat: false,
    sentiment: false,
    refresh: false,
  },
  setLoading: (key, value) => set((state) => ({
    loading: { ...state.loading, [key]: value }
  })),
  
  // UI state
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}))

export default useStore