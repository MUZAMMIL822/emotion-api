"""
main.py - v2
FastAPI backend with BERT emotion detection + Spotify song recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
import pandas as pd
import numpy as np
import os, sys, types, zipfile

# ── Emotion → Audio Feature Mapping ──────────────────────
EMOTION_FEATURES = {
    'admiration':    {'valence':(0.6,1.0),'energy':(0.4,0.8),'genres':['classical','piano','acoustic']},
    'amusement':     {'valence':(0.7,1.0),'energy':(0.5,0.9),'genres':['pop','comedy','children']},
    'anger':         {'valence':(0.0,0.3),'energy':(0.7,1.0),'genres':['metal','black-metal','alt-rock','rock']},
    'annoyance':     {'valence':(0.0,0.3),'energy':(0.6,1.0),'genres':['punk','grunge','alt-rock']},
    'approval':      {'valence':(0.6,1.0),'energy':(0.5,0.8),'genres':['pop','indie','happy']},
    'caring':        {'valence':(0.5,0.9),'energy':(0.2,0.5),'genres':['acoustic','folk','singer-songwriter']},
    'confusion':     {'valence':(0.3,0.6),'energy':(0.2,0.5),'genres':['ambient','chill','idm']},
    'curiosity':     {'valence':(0.4,0.7),'energy':(0.4,0.7),'genres':['jazz','bossa-nova','indie']},
    'desire':        {'valence':(0.5,0.8),'energy':(0.4,0.7),'genres':['r-n-b','soul','romance']},
    'disappointment':{'valence':(0.0,0.3),'energy':(0.1,0.4),'genres':['blues','sad','emo']},
    'disapproval':   {'valence':(0.1,0.4),'energy':(0.4,0.7),'genres':['alternative','grunge','punk']},
    'disgust':       {'valence':(0.0,0.3),'energy':(0.5,0.8),'genres':['metal','grunge','punk']},
    'embarrassment': {'valence':(0.3,0.6),'energy':(0.2,0.5),'genres':['indie','acoustic','folk']},
    'excitement':    {'valence':(0.7,1.0),'energy':(0.8,1.0),'genres':['edm','dance','party','electro']},
    'fear':          {'valence':(0.0,0.3),'energy':(0.3,0.6),'genres':['ambient','dark-techno','emo']},
    'gratitude':     {'valence':(0.7,1.0),'energy':(0.4,0.7),'genres':['gospel','soul','acoustic']},
    'grief':         {'valence':(0.0,0.2),'energy':(0.0,0.3),'genres':['classical','sad','blues']},
    'joy':           {'valence':(0.8,1.0),'energy':(0.7,1.0),'genres':['pop','happy','funk','dance']},
    'love':          {'valence':(0.6,1.0),'energy':(0.3,0.6),'genres':['romance','r-n-b','acoustic']},
    'nervousness':   {'valence':(0.2,0.5),'energy':(0.2,0.5),'genres':['ambient','chill','sleep']},
    'optimism':      {'valence':(0.7,1.0),'energy':(0.5,0.8),'genres':['indie','pop','folk']},
    'pride':         {'valence':(0.6,1.0),'energy':(0.7,1.0),'genres':['hip-hop','rap','work-out']},
    'realization':   {'valence':(0.4,0.7),'energy':(0.3,0.6),'genres':['post-rock','indie','alternative']},
    'relief':        {'valence':(0.5,0.8),'energy':(0.1,0.4),'genres':['acoustic','folk','chill']},
    'remorse':       {'valence':(0.0,0.3),'energy':(0.1,0.4),'genres':['blues','sad','acoustic']},
    'sadness':       {'valence':(0.0,0.3),'energy':(0.1,0.4),'genres':['sad','emo','blues','acoustic']},
    'surprise':      {'valence':(0.5,0.9),'energy':(0.6,0.9),'genres':['electronic','indie','pop']},
    'neutral':       {'valence':(0.3,0.7),'energy':(0.3,0.7),'genres':['chill','lo-fi','study']},
}

EMOTION_COLS = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise','neutral'
]
ID2LABEL = {i: e for i, e in enumerate(EMOTION_COLS)}
LABEL2ID = {e: i for i, e in enumerate(EMOTION_COLS)}

MODEL_PATH  = os.environ.get("MODEL_PATH", "./best_model")
DATASET_PATH = os.environ.get("DATASET_PATH", "./dataset.zip")
MAX_LEN     = 128
DROPOUT     = 0.3

# ── Fix __file__ issue ────────────────────────────────────
_mod = types.ModuleType("emotion_bert_module")
_mod.__file__ = "/tmp/emotion_bert_module.py"
sys.modules["emotion_bert_module"] = _mod
with open("/tmp/emotion_bert_module.py", "w") as f:
    f.write("# placeholder\n")

# ── Model ─────────────────────────────────────────────────
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
        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        logits = self.classifier(pooled)
        loss   = None
        return {'loss': loss, 'logits': logits}

EmotionBERT.__module__ = "emotion_bert_module"
_mod.EmotionBERT = EmotionBERT

# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(title="Emotion Music API v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device    = torch.device("cpu")
model     = None
tokenizer = None
music_df  = None

@app.on_event("startup")
async def load_all():
    global model, tokenizer, music_df

    # Load BERT model
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    config    = BertConfig.from_pretrained(MODEL_PATH, num_labels=len(EMOTION_COLS),
                    id2label=ID2LABEL, label2id=LABEL2ID)
    model = EmotionBERT.from_pretrained(MODEL_PATH, config=config, ignore_mismatched_sizes=True)
    model.eval().to(device)
    print("✅ BERT model loaded!")

    # Load Spotify dataset
    print("Loading Spotify dataset...")
    try:
        import zipfile
        with zipfile.ZipFile(DATASET_PATH) as z:
            with z.open("dataset.csv") as f:
                music_df = pd.read_csv(f)
        music_df = music_df.dropna(subset=['track_name','artists','valence','energy'])
        music_df = music_df[music_df['popularity'] > 20]
        music_df = music_df.drop_duplicates(subset=['track_name','artists'])
        print(f"✅ Spotify dataset loaded: {len(music_df):,} songs")
    except Exception as e:
        print(f"⚠️ Could not load Spotify dataset: {e}")

def get_songs(emotion: str, top_n: int = 5):
    if music_df is None: return []
    features = EMOTION_FEATURES.get(emotion, EMOTION_FEATURES['neutral'])
    val_min, val_max = features['valence']
    eng_min, eng_max = features['energy']
    genres = features['genres']

    filtered = music_df[
        (music_df['valence'] >= val_min) & (music_df['valence'] <= val_max) &
        (music_df['energy']  >= eng_min) & (music_df['energy']  <= eng_max)
    ]
    genre_match = filtered[filtered['track_genre'].isin(genres)]
    pool = genre_match if len(genre_match) >= top_n else filtered
    if len(pool) == 0: pool = music_df.sample(top_n)

    results = pool.sort_values('popularity', ascending=False).head(50).sample(min(top_n, len(pool)))
    return [{'track_name': r['track_name'], 'artists': r['artists'],
             'genre': r['track_genre'], 'popularity': int(r['popularity']),
             'valence': round(float(r['valence']),2), 'energy': round(float(r['energy']),2),
             'tempo': round(float(r['tempo']),1)} for _, r in results.iterrows()]

# ── Schemas ───────────────────────────────────────────────
class TextInput(BaseModel):
    text: str
    top_k: int = 3
    num_songs: int = 5

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Emotion Music API v2", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "songs_loaded": music_df is not None,
            "total_songs": len(music_df) if music_df is not None else 0}

@app.post("/predict")
def predict(input: TextInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    enc = tokenizer(input.text, padding='max_length', truncation=True,
                    max_length=MAX_LEN, return_tensors='pt')
    with torch.no_grad():
        out   = model(input_ids=enc['input_ids'].to(device),
                      attention_mask=enc['attention_mask'].to(device))
        probs = F.softmax(out['logits'], dim=-1).squeeze()
        top_p, top_i = torch.topk(probs, k=min(input.top_k, len(EMOTION_COLS)))

    top_emotions = [{"emotion": ID2LABEL[i.item()], "confidence": round(p.item(), 4)}
                    for i, p in zip(top_i, top_p)]
    top_emotion = top_emotions[0]["emotion"]
    songs = get_songs(top_emotion, input.num_songs)

    return {
        "text":        input.text,
        "top_emotion": top_emotion,
        "confidence":  top_emotions[0]["confidence"],
        "top_k":       top_emotions,
        "songs":       songs
    }
