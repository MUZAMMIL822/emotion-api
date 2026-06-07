"""
main.py - v6 (chatbot + emotion + songs)
Endpoints:
  GET  /health
  POST /predict          — direct text → emotion + songs
  POST /chat/analyze     — 5 answers → summary → emotion + songs
  GET  /more/{emotion}   — more songs for emotion
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch, torch.nn.functional as F
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
import os, sys, types, json, random, gc

# ── Emotion config ────────────────────────────────────────
EMOTION_COLS = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise','neutral'
]
ID2LABEL = {i: e for i, e in enumerate(EMOTION_COLS)}
LABEL2ID = {e: i for i, e in enumerate(EMOTION_COLS)}

# ── Indirect questions ────────────────────────────────────
CHAT_QUESTIONS = [
    "What was the highlight of your day today?",
    "If your day was a type of weather, what would it be and why?",
    "What is the last thing that made you smile or sigh today?",
    "If you could do just one thing right now, what would it be?",
    "Describe your day in just three words.",
]

MODEL_PATH = os.environ.get("MODEL_PATH", "./best_model")
SONGS_PATH = os.environ.get("SONGS_PATH", "./songs_lookup.json")
MAX_LEN    = 128
DROPOUT    = 0.3

# ── Fix Colab __file__ issue ──────────────────────────────
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

# ── App ───────────────────────────────────────────────────
app = FastAPI(title="Emotion Music API", version="6.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device       = torch.device("cpu")
model        = None
tokenizer    = None
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
    try:
        with open(SONGS_PATH, 'r') as f:
            songs_lookup = json.load(f)
        print(f"✅ Songs loaded for {len(songs_lookup)} emotions")
    except Exception as e:
        print(f"⚠️ Songs load failed: {e}")

# ── Helper: predict emotion from text ────────────────────
def predict_emotion(text: str, top_k: int = 3):
    enc = tokenizer(text, padding='max_length', truncation=True,
                    max_length=MAX_LEN, return_tensors='pt')
    with torch.no_grad():
        out   = model(input_ids=enc['input_ids'].to(device),
                      attention_mask=enc['attention_mask'].to(device))
        probs = F.softmax(out['logits'], dim=-1).squeeze()
        top_p, top_i = torch.topk(probs, k=min(top_k, len(EMOTION_COLS)))
    return [{"emotion": ID2LABEL[i.item()], "confidence": round(p.item(), 4)}
            for i, p in zip(top_i, top_p)]

# ── Helper: build weighted summary ───────────────────────
def build_summary(answers: List[str]) -> str:
    """
    Build weighted summary from 5 answers.
    Later questions reveal emotion more directly so weighted more.
    Q1=1x, Q2=1x, Q3=2x, Q4=2x, Q5=3x
    Also prepend question context so BERT understands the answer better.
    """
    weights = [1, 1, 2, 2, 3]
    contexts = [
        "The highlight of my day was",
        "My day felt like",
        "The last thing that made me feel was",
        "Right now I want to",
        "My day was",
    ]
    parts = []
    for ans, w, ctx in zip(answers, weights, contexts):
        if ans.strip():
            # Prepend context to help BERT understand the answer
            enriched = f"{ctx} {ans.strip()}"
            parts.extend([enriched] * w)
    return '. '.join(parts)

# ── Emotion → ideal audio profile ────────────────────────
EMOTION_AUDIO = {
    'joy':            {'valence': 0.8, 'energy': 0.7},
    'excitement':     {'valence': 0.7, 'energy': 0.9},
    'love':           {'valence': 0.8, 'energy': 0.5},
    'admiration':     {'valence': 0.7, 'energy': 0.5},
    'amusement':      {'valence': 0.8, 'energy': 0.6},
    'gratitude':      {'valence': 0.7, 'energy': 0.4},
    'optimism':       {'valence': 0.75,'energy': 0.6},
    'pride':          {'valence': 0.7, 'energy': 0.6},
    'relief':         {'valence': 0.6, 'energy': 0.3},
    'caring':         {'valence': 0.65,'energy': 0.4},
    'sadness':        {'valence': 0.2, 'energy': 0.3},
    'grief':          {'valence': 0.1, 'energy': 0.2},
    'disappointment': {'valence': 0.2, 'energy': 0.3},
    'remorse':        {'valence': 0.2, 'energy': 0.2},
    'anger':          {'valence': 0.2, 'energy': 0.9},
    'annoyance':      {'valence': 0.3, 'energy': 0.7},
    'disgust':        {'valence': 0.2, 'energy': 0.6},
    'fear':           {'valence': 0.2, 'energy': 0.6},
    'nervousness':    {'valence': 0.3, 'energy': 0.6},
    'confusion':      {'valence': 0.4, 'energy': 0.5},
    'curiosity':      {'valence': 0.6, 'energy': 0.6},
    'surprise':       {'valence': 0.6, 'energy': 0.7},
    'desire':         {'valence': 0.6, 'energy': 0.5},
    'embarrassment':  {'valence': 0.3, 'energy': 0.4},
    'realization':    {'valence': 0.5, 'energy': 0.4},
    'approval':       {'valence': 0.65,'energy': 0.5},
    'disapproval':    {'valence': 0.3, 'energy': 0.5},
    'neutral':        {'valence': 0.5, 'energy': 0.5},
}

# ── Helper: get songs ranked by audio feature match ──────
def get_songs(emotion: str, num: int = 10, section: str = "recommended"):
    data  = songs_lookup.get(emotion, songs_lookup.get('neutral', {}))
    songs = data.get(section, data.get('recommended', [])) if isinstance(data, dict) else data
    if not songs: return []

    # Get ideal audio profile for this emotion
    profile = EMOTION_AUDIO.get(emotion, {'valence': 0.5, 'energy': 0.5})
    tv = profile['valence']
    te = profile['energy']

    # Score each song by distance from ideal profile
    # Also factor in popularity for tie-breaking
    def score(s):
        v = float(s.get('valence', 0.5))
        e = float(s.get('energy',  0.5))
        p = float(s.get('popularity', 50)) / 100.0
        # Lower distance = better match
        distance = ((v - tv) ** 2 + (e - te) ** 2) ** 0.5
        return distance - (p * 0.05)  # popularity gives slight boost

    # Sort by best match, add small randomness in top candidates
    # to avoid always returning exact same songs
    sorted_songs = sorted(songs, key=score)
    # Take top 2x candidates and randomly sample from those
    top_pool = sorted_songs[:min(num * 2, len(sorted_songs))]
    selected = random.sample(top_pool, min(num, len(top_pool)))

    return [{'track_name': s['track_name'], 'artists': s['artists'],
             'genre': s['track_genre'], 'popularity': int(s['popularity']),
             'valence': round(float(s['valence']), 2),
             'energy':  round(float(s['energy']),  2),
             'tempo':   round(float(s['tempo']),   1)} for s in selected]

# ── Models ────────────────────────────────────────────────
class TextInput(BaseModel):
    text:      str
    top_k:     int = 3
    num_songs: int = 10

class ChatInput(BaseModel):
    answers:   List[str]  # list of 5 answers
    top_k:     int = 3
    num_songs: int = 10

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Emotion Music API", "version": "6.0.0"}

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model_loaded":   model is not None,
        "songs_loaded":   len(songs_lookup) > 0,
        "total_emotions": len(songs_lookup),
        "version":        "6.0.0"
    }

@app.get("/questions")
def get_questions():
    """Return the 5 indirect chat questions."""
    return {"questions": CHAT_QUESTIONS}

@app.post("/predict")
def predict(input: TextInput):
    """Direct text → emotion + songs."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    top_emotions = predict_emotion(input.text, input.top_k)
    top_emotion  = top_emotions[0]["emotion"]

    return {
        "text":        input.text,
        "top_emotion": top_emotion,
        "confidence":  top_emotions[0]["confidence"],
        "top_k":       top_emotions,
        "songs":       get_songs(top_emotion, input.num_songs, "recommended"),
        "more_songs":  get_songs(top_emotion, 10, "more"),
    }

@app.post("/chat/analyze")
def chat_analyze(input: ChatInput):
    """
    5 indirect answers → weighted summary → emotion + songs.
    This is the main endpoint for the chatbot flow.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not input.answers:
        raise HTTPException(status_code=400, detail="Answers are empty")

    # Build weighted summary
    summary = build_summary(input.answers)
    if not summary.strip():
        raise HTTPException(status_code=400, detail="All answers are empty")

    # Detect emotion
    top_emotions = predict_emotion(summary, input.top_k)
    top_emotion  = top_emotions[0]["emotion"]

    return {
        "answers":     input.answers,
        "summary":     summary,
        "top_emotion": top_emotion,
        "confidence":  top_emotions[0]["confidence"],
        "top_k":       top_emotions,
        "songs":       get_songs(top_emotion, input.num_songs, "recommended"),
        "more_songs":  get_songs(top_emotion, 10, "more"),
    }

@app.get("/more/{emotion}")
def more_songs(emotion: str, limit: int = 10):
    return {"emotion": emotion, "songs": get_songs(emotion, limit, "more")}
