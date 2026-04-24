import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [spokenText, setSpokenText] = useState('')
  const [analysis, setAnalysis] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleStart = async () => {
    setLoading(true)
    setError('')
    setSpokenText('')
    setAnalysis('')

    try {
      const response = await axios.post('/api/analyze')
      setSpokenText(response.data.spoken_text)
      setAnalysis(response.data.result)
    } catch (err) {
      setError(err.response?.data?.error || 'เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>🧠 MentalHack</h1>
        <p>วิเคราะห์สุขภาพจิตจากเสียงของคุณ</p>
      </header>

      <main className="main-content">
        <button
          onClick={handleStart}
          disabled={loading}
          className={`start-btn ${loading ? 'loading' : ''}`}
        >
          {loading ? 'กำลังประมวลผล...' : '🎤 เริ่ม'}
        </button>

        {error && (
          <div className="error-message">
            <p>❌ {error}</p>
          </div>
        )}

        {spokenText && (
          <section className="result-section">
            <h2>คุณพูดว่า</h2>
            <div className="result-content">
              <p>{spokenText}</p>
            </div>
          </section>
        )}

        {analysis && (
          <section className="result-section">
            <h2>ผลการวิเคราะห์</h2>
            <div className="result-content analysis">
              <p>{analysis}</p>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>⚠️ แอปพลิเคชันนี้ใช้สำหรับการวิเคราะห์เบื้องต้นเท่านั้น หากมีปัญหาจิตใจ กรุณาติดต่อผู้เชี่ยวชาญ</p>
      </footer>
    </div>
  )
}

export default App
