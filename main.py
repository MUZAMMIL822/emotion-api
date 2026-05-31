"""
main.py - v4 (ultra memory optimized — uses pre-built song lookup JSON)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, torch.nn.functional as F
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
import os, sys, types, json, random, gc

EMOTION_COLS = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise','neutral'
]
ID2LABEL   = {i: e for i, e in enumerate(EMOTION_COLS)}
LABEL2ID   = {e: i for i, e in enumerate(EMOTION_COLS)}
MODEL_PATH = os.environ.get("MODEL_PATH",  "./best_model")
SONGS_PATH = os.environ.get("SONGS_PATH",  "./songs_lookup.json")
MAX_LEN    = 128
DROPOUT    = 0.3

# ── Fix __file__ ──────────────────────────────────────────
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
        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        logits = self.classifier(pooled)
        return {'logits': logits}

EmotionBERT.__module__ = "emotion_bert_module"
_mod.EmotionBERT = EmotionBERT

app = FastAPI(title="Emotion Music API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device      = torch.device("cpu")
model       = None
tokenizer   = None
songs_lookup = {}

@app.on_event("startup")
async def load_all():
    global model, tokenizer, songs_lookup

    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    config    = BertConfig.from_pretrained(MODEL_PATH,
                    num_labels=len(EMOTION_COLS),
                    id2label=ID2LABEL, label2id=LABEL2ID)
    model = EmotionBERT.from_pretrained(MODEL_PATH, config=config,
                ignore_mismatched_sizes=True)
    model.eval().to(device)
    gc.collect()
    print("✅ BERT loaded!")

    print("Loading songs lookup...")
    try:
        with open(SONGS_PATH, 'r') as f:
            songs_lookup = json.load(f)
        total = sum(len(v) for v in songs_lookup.values())
        print(f"✅ Songs loaded: {total} songs for {len(songs_lookup)} emotions")
    except Exception as e:
        print(f"⚠️ Songs load failed: {e}")

def get_songs(emotion: str, top_n: int = 5):
    songs = songs_lookup.get(emotion, songs_lookup.get('neutral', []))
    if not songs: return []
    sample = random.sample(songs, min(top_n, len(songs)))
    return [{'track_name': s['track_name'], 'artists': s['artists'],
             'genre': s['track_genre'], 'popularity': int(s['popularity']),
             'valence': round(float(s['valence']),2),
             'energy': round(float(s['energy']),2),
             'tempo': round(float(s['tempo']),1)} for s in sample]

class TextInput(BaseModel):
    text: str
    top_k: int = 3
    num_songs: int = 5

@app.get("/")
def root():
    return {"message": "Emotion Music API", "version": "4.0.0"}

@app.get("/health")
def health():
    total = sum(len(v) for v in songs_lookup.values())
    return {"status": "ok", "model_loaded": model is not None,
            "songs_loaded": len(songs_lookup) > 0, "total_songs": total}

@app.get("/debug")
def debug():
    return {"files": os.listdir("."), "songs_path_exists": os.path.exists(SONGS_PATH),
            "model_path_exists": os.path.exists(MODEL_PATH), "cwd": os.getcwd()}

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
    top_emotion  = top_emotions[0]["emotion"]
    songs        = get_songs(top_emotion, input.num_songs)

    return {"text": input.text, "top_emotion": top_emotion,
            "confidence": top_emotions[0]["confidence"],
            "top_k": top_emotions, "songs": songs}
