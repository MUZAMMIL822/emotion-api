# Emotion Music API — FastAPI Backend

## 📁 Project Structure
```
emotion_api/
├── main.py           ← FastAPI app + model inference
├── best_model/       ← Copy your trained model here
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── requirements.txt
├── Procfile
└── railway.toml
```

---

## 🚀 Deploy to Railway

### Step 1 — Copy your trained model
Download `best_model/` folder from Google Drive and place it here.

### Step 2 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial API commit"
git remote add origin https://github.com/YOUR_USERNAME/emotion-api.git
git push -u origin main
```

### Step 3 — Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. Click **New Project → Deploy from GitHub**
3. Select your repo
4. Railway auto-detects and deploys ✅

### Step 4 — Get your API URL
Railway gives you a URL like:
```
https://emotion-api-production.up.railway.app
```

---

## 📡 API Endpoints

### POST /predict
```json
Request:
{
  "text": "I just got promoted! Best day ever!",
  "top_k": 3
}

Response:
{
  "text": "I just got promoted! Best day ever!",
  "top_emotion": "joy",
  "confidence": 0.87,
  "top_k": [
    {"emotion": "joy",       "confidence": 0.87},
    {"emotion": "excitement","confidence": 0.08},
    {"emotion": "pride",     "confidence": 0.03}
  ],
  "music": {
    "genre": "Pop / Funk",
    "mood":  "Happy",
    "tempo": "Fast"
  }
}
```

### GET /health
```json
{"status": "ok", "model_loaded": true}
```

### GET /emotions
```json
{"emotions": ["admiration", "amusement", ...], "total": 28}
```

---

## 🧪 Test Locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Visit: http://localhost:8000/docs
```
