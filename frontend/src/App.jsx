import { useState } from 'react'
import { recordAndTranscribe } from './services/audioService'
import { analyzeWithGroq } from './services/groqService'
import './App.css'

function App() {
  const [spokenText, setSpokenText] = useState('')
  const [analysis, setAnalysis] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [step, setStep] = useState('')

  const handleStart = async () => {
    setLoading(true)
    setError('')
    setSpokenText('')
    setAnalysis('')
    setStep('')

    try {
      setStep('🎤 กำลังฟังเสียง...')
      const text = await recordAndTranscribe()
      setSpokenText(text)

      setStep('🤖 กำลังวิเคราะห์...')
      const result = await analyzeWithGroq(text)
      setAnalysis(result)
    } catch (err) {
      setError(err.message || 'เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง')
      console.error('Error:', err)
    } finally {
      setLoading(false)
      setStep('')
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
          {loading ? '⏳ กำลังประมวลผล...' : '🎤 เริ่มพูด'}
        </button>

        {step && (
          <div className="step-indicator">
            <p>{step}</p>
          </div>
        )}

        {error && (
          <div className="error-message">
            <p>❌ {error}</p>
          </div>
        )}

        {spokenText && (
          <section className="result-section">
            <h2>📝 คุณพูดว่า</h2>
            <div className="result-content">
              <p>{spokenText}</p>
            </div>
          </section>
        )}

        {analysis && (
          <section className="result-section">
            <h2>🔍 ผลการวิเคราะห์</h2>
            <div className="result-content analysis">
              <p>{analysis}</p>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>⚠️ แอปพลิเคชันนี้ใช้สำหรับการวิเคราะห์เบื้องต้นเท่านั้น หากมีปัญหาจิตใจรุนแรง กรุณาติดต่อผู้เชี่ยวชาญทันที</p>
        <p style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>Powered by Groq AI ⚡</p>
      </footer>
    </div>
  )
}

export default App
