# 🧠 MentalHack

แอปพลิเคชันวิเคราะห์สุขภาพจิตจากเสียงของคุณ ใช้เทคโนโลยี AI และ Speech Recognition

## 🚀 Features

- 🎤 **บันทึกเสียง** - ฟังเสียงของคุณจากไมค์
- 🔊 **แปลเป็นข้อความ** - แปลงเสียงเป็นข้อความภาษาไทยโดยอัตโนมัติ
- 🤖 **วิเคราะห์ AI** - ใช้ Google Gemma เพื่อวิเคราะห์สถานะสุขภาพจิต
- ✨ **UI Modern** - ออกแบบด้วย React ที่สวยงาม

## 🏗️ Project Structure

```
mentalhack/
├── backend/
│   ├── app.py                 # Flask server
│   ├── requirements.txt       # Python dependencies
│   └── .env.example          # Environment variables template
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main React component
    │   ├── App.css           # Styling
    │   ├── index.css         # Global styles
    │   └── main.jsx          # Entry point
    ├── index.html            # HTML template
    ├── package.json          # NPM dependencies
    ├── vite.config.js        # Vite configuration
    └── .env.example          # Environment variables template
```

## 📋 Requirements

- Python 3.8+
- Node.js 16+ & npm
- Microphone for audio input
- HuggingFace API key

## 🔧 Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd mentalhack
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Add your HuggingFace API key to .env
# HUGGINGFACE_API_KEY=your_key_here
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env
```

## 🚀 Running the Application

### Option 1: Separate Terminals

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

### Option 2: Docker Compose (Recommended)

```bash
docker-compose up
```

Both services will start automatically:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5000`

## 🌐 Deployment

### Frontend (Vercel/Netlify)
```bash
cd frontend
npm run build
```

Then deploy the `dist` folder to Vercel or Netlify.

### Backend (Heroku/Railway)

1. Create a `Procfile`:
```
web: cd backend && gunicorn app:app
```

2. Add to `requirements.txt`: `gunicorn==20.1.0`

3. Deploy to Heroku:
```bash
git push heroku main
```

## 🔒 Security Notes

- ✅ API key stored in environment variables (never committed)
- ✅ CORS enabled for cross-origin requests
- ✅ Debug mode disabled in production
- ✅ Error handling for all API endpoints

## 📚 API Endpoints

- `POST /api/analyze` - Process audio and return analysis
  - Response: `{ spoken_text: string, result: string }`
  - Error: `{ error: string }`

- `GET /api/health` - Health check
  - Response: `{ status: "ok" }`

## ⚠️ Disclaimer

แอปพลิเคชันนี้ใช้สำหรับการวิเคราะห์เบื้องต้นเท่านั้น หากคุณมีปัญหาสุขภาพจิตที่ร้ายแรง กรุณาติดต่อผู้เชี่ยวชาญหรือโทรติดต่อสายด่วนสุขภาพจิต

## 🛠️ Environment Variables

### Backend (.env)
```
HUGGINGFACE_API_KEY=your_key_here
MIC_INDEX=1
FLASK_DEBUG=False
```

### Frontend (.env)
```
VITE_API_URL=http://localhost:5000
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use this project as you wish.
