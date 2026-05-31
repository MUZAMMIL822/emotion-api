"""
main.py
FastAPI backend — Emotion-Aware Music Recommendation System
Serves trained EmotionBERT model as a REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
import os, sys, types

# ── Config ────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH", "./best_model")
MAX_LEN     = 128
DROPOUT     = 0.3

EMOTION_COLS = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise','neutral'
]
ID2LABEL = {i: e for i, e in enumerate(EMOTION_COLS)}
LABEL2ID = {e: i for i, e in enumerate(EMOTION_COLS)}

EMOTION_TO_MUSIC = {
    'admiration':    {'genre':'Classical',      'mood':'Inspiring',   'tempo':'Moderate'},
    'amusement':     {'genre':'Pop',            'mood':'Playful',     'tempo':'Upbeat'},
    'anger':         {'genre':'Metal / Rock',   'mood':'Intense',     'tempo':'Fast'},
    'annoyance':     {'genre':'Punk',           'mood':'Aggressive',  'tempo':'Fast'},
    'approval':      {'genre':'Pop',            'mood':'Positive',    'tempo':'Moderate'},
    'caring':        {'genre':'Acoustic',       'mood':'Warm',        'tempo':'Slow'},
    'confusion':     {'genre':'Ambient',        'mood':'Dreamy',      'tempo':'Slow'},
    'curiosity':     {'genre':'Jazz',           'mood':'Exploratory', 'tempo':'Moderate'},
    'desire':        {'genre':'R&B / Soul',     'mood':'Romantic',    'tempo':'Moderate'},
    'disappointment':{'genre':'Blues',          'mood':'Melancholic', 'tempo':'Slow'},
    'disapproval':   {'genre':'Alternative',    'mood':'Brooding',    'tempo':'Moderate'},
    'disgust':       {'genre':'Grunge',         'mood':'Dark',        'tempo':'Moderate'},
    'embarrassment': {'genre':'Indie Pop',      'mood':'Quirky',      'tempo':'Moderate'},
    'excitement':    {'genre':'EDM / Dance',    'mood':'Energetic',   'tempo':'Fast'},
    'fear':          {'genre':'Cinematic',      'mood':'Tense',       'tempo':'Slow'},
    'gratitude':     {'genre':'Gospel / Soul',  'mood':'Uplifting',   'tempo':'Moderate'},
    'grief':         {'genre':'Classical',      'mood':'Sorrowful',   'tempo':'Very Slow'},
    'joy':           {'genre':'Pop / Funk',     'mood':'Happy',       'tempo':'Fast'},
    'love':          {'genre':'Romantic Pop',   'mood':'Tender',      'tempo':'Slow'},
    'nervousness':   {'genre':'Ambient',        'mood':'Calming',     'tempo':'Slow'},
    'optimism':      {'genre':'Indie Pop',      'mood':'Hopeful',     'tempo':'Upbeat'},
    'pride':         {'genre':'Hip-Hop',        'mood':'Confident',   'tempo':'Moderate'},
    'realization':   {'genre':'Post-Rock',      'mood':'Reflective',  'tempo':'Moderate'},
    'relief':        {'genre':'Acoustic/Folk',  'mood':'Peaceful',    'tempo':'Slow'},
    'remorse':       {'genre':'Blues',          'mood':'Regretful',   'tempo':'Slow'},
    'sadness':       {'genre':'Sad Indie',      'mood':'Melancholic', 'tempo':'Slow'},
    'surprise':      {'genre':'Electronic',     'mood':'Unexpected',  'tempo':'Varied'},
    'neutral':       {'genre':'Lo-fi / Chill',  'mood':'Balanced',    'tempo':'Moderate'},
}

# ── Model Definition ──────────────────────────────────────
# Fix for Railway/production: __file__ module trick
_mod = types.ModuleType("emotion_bert_module")
_mod.__file__ = "/tmp/emotion_bert_module.py"
sys.modules["emotion_bert_module"] = _mod
with open("/tmp/emotion_bert_module.py", "w") as f:
    f.write("# placeholder\n")

class EmotionBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, config.num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        logits = self.classifier(pooled)
        loss   = self.loss_fn(logits, labels) if labels is not None else None
        return {'loss': loss, 'logits': logits}

EmotionBERT.__module__ = "emotion_bert_module"
_mod.EmotionBERT = EmotionBERT

# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="Emotion-Aware Music API",
    description="Detects emotion from text and recommends music genre/mood",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model on Startup ─────────────────────────────────
device    = torch.device("cpu")  # Railway free tier = CPU
model     = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    config    = BertConfig.from_pretrained(MODEL_PATH,
                    num_labels=len(EMOTION_COLS),
                    id2label=ID2LABEL, label2id=LABEL2ID)
    model = EmotionBERT.from_pretrained(MODEL_PATH, config=config,
                ignore_mismatched_sizes=True)
    model.eval()
    model.to(device)
    print("✅ Model loaded successfully!")

# ── Request / Response Schemas ────────────────────────────
class TextInput(BaseModel):
    text: str
    top_k: int = 3

class EmotionResult(BaseModel):
    text: str
    top_emotion: str
    confidence: float
    top_k: list
    music: dict

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Emotion-Aware Music Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET  /health",
            "emotions":"GET  /emotions"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/emotions")
def get_emotions():
    return {"emotions": EMOTION_COLS, "total": len(EMOTION_COLS)}

@app.post("/predict", response_model=EmotionResult)
def predict(input: TextInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    enc = tokenizer(
        input.text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )

    with torch.no_grad():
        out   = model(input_ids=enc['input_ids'].to(device),
                      attention_mask=enc['attention_mask'].to(device))
        probs = F.softmax(out['logits'], dim=-1).squeeze()
        top_p, top_i = torch.topk(probs, k=min(input.top_k, len(EMOTION_COLS)))

    top_emotions = [
        {"emotion": ID2LABEL[i.item()], "confidence": round(p.item(), 4)}
        for i, p in zip(top_i, top_p)
    ]
    top_emotion = top_emotions[0]["emotion"]
    confidence  = top_emotions[0]["confidence"]
    music       = EMOTION_TO_MUSIC.get(top_emotion, {})

    return EmotionResult(
        text=input.text,
        top_emotion=top_emotion,
        confidence=confidence,
        top_k=top_emotions,
        music=music
    )
