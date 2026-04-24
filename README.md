# 🧠 MentalHack

แอปพลิเคชันวิเคราะห์สุขภาพจิตจากเสียงของคุณ ใช้เทคโนโลยี AI (Groq) และ Web Speech API

**ไม่ต้องมี Backend! Frontend เพียงอย่างเดียว** 🚀

## 🚀 Features

- 🎤 **บันทึกเสียง** - ฟังเสียงจากไมค์ด้วย Web Audio API
- 🔊 **แปลเป็นข้อความ** - แปลงเสียงเป็นข้อความภาษาไทยโดยอัตโนมัติ
- 🤖 **วิเคราะห์ AI** - ใช้ Groq AI (ฟรี + เร็วมาก) เพื่อวิเคราะห์สุขภาพจิต
- ✨ **UI Modern** - ออกแบบด้วย React + Vite ที่สวยงาม

## 🏗️ Project Structure

```
mentalhack/frontend/
├── src/
│   ├── App.jsx                    # Main component
│   ├── App.css                    # Styling
│   ├── services/
│   │   ├── groqService.js        # Groq API integration
│   │   └── audioService.js       # Audio recording + speech-to-text
│   ├── index.css                 # Global styles
│   └── main.jsx                  # Entry point
├── index.html                     # HTML template
├── package.json                   # NPM dependencies
├── vite.config.js                 # Vite configuration
├── vercel.json                    # Vercel deployment config
└── .env.example                   # Environment variables template
```

## 📋 Requirements

- Node.js 16+ & npm
- Microphone for audio input
- Groq API key (ฟรี)

## 🔧 Setup

### 1. Get Groq API Key (ฟรี)
1. ไปที่ https://console.groq.com/
2. สร้าง account และ log in
3. สร้าง API key ใหม่
4. Copy key มาไว้

### 2. Clone & Setup
```bash
git clone <repository-url>
cd mentalhack/frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Add your Groq API key
echo "VITE_GROQ_API_KEY=your_groq_key_here" > .env
```

## 🚀 Running Locally

```bash
cd frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

## 🌐 Deploy to Vercel (ฟรี)

### Automatic (Best)
1. Push code to GitHub
2. ไปที่ https://vercel.com/new
3. Connect GitHub repository
4. Set **Root Directory** to `frontend`
5. Add Environment Variable:
   - Key: `VITE_GROQ_API_KEY`
   - Value: `your_groq_api_key`
6. Click Deploy ✅

### Manual
```bash
npm run build
# Deploy dist/ folder to Vercel
```

## 🔒 Security Notes

- ✅ API key stored in environment variables only
- ✅ Never commit .env file
- ✅ Groq API key has rate limiting (ฟรี tier)
- ✅ Web Audio API requires HTTPS in production

## 🎯 How It Works

1. **User speaks** → Web Audio API records audio
2. **Speech to Text** → Web Speech API (Thai) converts to text
3. **Send to Groq** → Groq API analyzes the text
4. **Display results** → Shows analysis in Thai

## 📝 Technology Stack

- **Frontend**: React 18 + Vite
- **Speech Recognition**: Web Speech API
- **Audio Recording**: MediaRecorder API
- **AI/LLM**: Groq API (Mixtral 8x7b)
- **Styling**: Pure CSS

## ⚠️ Disclaimer

แอปพลิเคชันนี้ใช้สำหรับการวิเคราะห์เบื้องต้นเท่านั้น หากคุณมีปัญหาสุขภาพจิตที่ร้ายแรง กรุณาติดต่อผู้เชี่ยวชาญทันที

## 🛠️ Environment Variables

```
VITE_GROQ_API_KEY=your_groq_api_key_here
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use this project as you wish.

---

**Questions?** ติดต่อสอบถามได้เลยครับ! 🙌
