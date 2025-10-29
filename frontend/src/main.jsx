import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1a1f3a',
            color: '#e0e6ed',
            border: '1px solid rgba(102, 126, 234, 0.3)',
          },
          success: {
            iconTheme: {
              primary: '#28a745',
              secondary: '#1a1f3a',
            },
          },
          error: {
            iconTheme: {
              primary: '#dc3545',
              secondary: '#1a1f3a',
            },
          },
        }}
      />
    </BrowserRouter>
  </React.StrictMode>,
)