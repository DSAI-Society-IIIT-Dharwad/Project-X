import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ChatAssistant from './pages/ChatAssistant'
import SentimentAnalysis from './pages/SentimentAnalysis'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/chat" element={<ChatAssistant />} />
        <Route path="/sentiment" element={<SentimentAnalysis />} />
      </Routes>
    </Layout>
  )
}

export default App