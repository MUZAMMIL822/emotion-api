"""
music_recommender.py
Emotion → Audio Features → Spotify Song Recommendations
"""

import pandas as pd
import numpy as np
import zipfile, io, os

# ── Emotion → Audio Feature Mapping ──────────────────────
EMOTION_FEATURES = {
    'admiration':    {'valence':(0.6,1.0), 'energy':(0.4,0.8), 'genres':['classical','piano','acoustic']},
    'amusement':     {'valence':(0.7,1.0), 'energy':(0.5,0.9), 'genres':['pop','comedy','children']},
    'anger':         {'valence':(0.0,0.3), 'energy':(0.7,1.0), 'genres':['metal','black-metal','alt-rock','rock']},
    'annoyance':     {'valence':(0.0,0.3), 'energy':(0.6,1.0), 'genres':['punk','grunge','alt-rock']},
    'approval':      {'valence':(0.6,1.0), 'energy':(0.5,0.8), 'genres':['pop','indie','happy']},
    'caring':        {'valence':(0.5,0.9), 'energy':(0.2,0.5), 'genres':['acoustic','folk','singer-songwriter']},
    'confusion':     {'valence':(0.3,0.6), 'energy':(0.2,0.5), 'genres':['ambient','chill','idm']},
    'curiosity':     {'valence':(0.4,0.7), 'energy':(0.4,0.7), 'genres':['jazz','bossa-nova','indie']},
    'desire':        {'valence':(0.5,0.8), 'energy':(0.4,0.7), 'genres':['r-n-b','soul','romance']},
    'disappointment':{'valence':(0.0,0.3), 'energy':(0.1,0.4), 'genres':['blues','sad','emo']},
    'disapproval':   {'valence':(0.1,0.4), 'energy':(0.4,0.7), 'genres':['alternative','grunge','punk']},
    'disgust':       {'valence':(0.0,0.3), 'energy':(0.5,0.8), 'genres':['metal','grunge','punk']},
    'embarrassment': {'valence':(0.3,0.6), 'energy':(0.2,0.5), 'genres':['indie','acoustic','folk']},
    'excitement':    {'valence':(0.7,1.0), 'energy':(0.8,1.0), 'genres':['edm','dance','party','electro']},
    'fear':          {'valence':(0.0,0.3), 'energy':(0.3,0.6), 'genres':['ambient','dark-techno','emo']},
    'gratitude':     {'valence':(0.7,1.0), 'energy':(0.4,0.7), 'genres':['gospel','soul','acoustic']},
    'grief':         {'valence':(0.0,0.2), 'energy':(0.0,0.3), 'genres':['classical','sad','blues']},
    'joy':           {'valence':(0.8,1.0), 'energy':(0.7,1.0), 'genres':['pop','happy','funk','dance']},
    'love':          {'valence':(0.6,1.0), 'energy':(0.3,0.6), 'genres':['romance','r-n-b','acoustic']},
    'nervousness':   {'valence':(0.2,0.5), 'energy':(0.2,0.5), 'genres':['ambient','chill','sleep']},
    'optimism':      {'valence':(0.7,1.0), 'energy':(0.5,0.8), 'genres':['indie','pop','folk']},
    'pride':         {'valence':(0.6,1.0), 'energy':(0.7,1.0), 'genres':['hip-hop','rap','work-out']},
    'realization':   {'valence':(0.4,0.7), 'energy':(0.3,0.6), 'genres':['post-rock','indie','alternative']},
    'relief':        {'valence':(0.5,0.8), 'energy':(0.1,0.4), 'genres':['acoustic','folk','chill']},
    'remorse':       {'valence':(0.0,0.3), 'energy':(0.1,0.4), 'genres':['blues','sad','acoustic']},
    'sadness':       {'valence':(0.0,0.3), 'energy':(0.1,0.4), 'genres':['sad','emo','blues','acoustic']},
    'surprise':      {'valence':(0.5,0.9), 'energy':(0.6,0.9), 'genres':['electronic','indie','pop']},
    'neutral':       {'valence':(0.3,0.7), 'energy':(0.3,0.7), 'genres':['chill','lo-fi','study']},
}

class MusicRecommender:
    def __init__(self, data_path):
        print("Loading Spotify dataset...")
        if data_path.endswith('.zip'):
            with zipfile.ZipFile(data_path, 'r') as z:
                with z.open('dataset.csv') as f:
                    self.df = pd.read_csv(f)
        else:
            self.df = pd.read_csv(data_path)

        # Clean dataset
        self.df = self.df.dropna(subset=['track_name','artists','valence','energy'])
        self.df = self.df[self.df['popularity'] > 20]  # only popular songs
        self.df = self.df.drop_duplicates(subset=['track_name','artists'])
        print(f"✅ Loaded {len(self.df):,} songs across {self.df['track_genre'].nunique()} genres")

    def recommend(self, emotion: str, top_n: int = 5) -> list:
        emotion = emotion.lower()
        if emotion not in EMOTION_FEATURES:
            emotion = 'neutral'

        features = EMOTION_FEATURES[emotion]
        val_min, val_max = features['valence']
        eng_min, eng_max = features['energy']
        target_genres   = features['genres']

        # Filter by audio features
        filtered = self.df[
            (self.df['valence'] >= val_min) & (self.df['valence'] <= val_max) &
            (self.df['energy']  >= eng_min) & (self.df['energy']  <= eng_max)
        ].copy()

        # Boost songs from matching genres
        genre_match = filtered[filtered['track_genre'].isin(target_genres)]

        # Use genre match if enough songs, else use audio feature match
        pool = genre_match if len(genre_match) >= top_n else filtered

        if len(pool) == 0:
            pool = self.df.sample(top_n)

        # Sort by popularity and sample
        pool = pool.sort_values('popularity', ascending=False)
        results = pool.head(50).sample(min(top_n, len(pool)))

        songs = []
        for _, row in results.iterrows():
            songs.append({
                'track_name':  row['track_name'],
                'artists':     row['artists'],
                'genre':       row['track_genre'],
                'popularity':  int(row['popularity']),
                'valence':     round(float(row['valence']), 2),
                'energy':      round(float(row['energy']), 2),
                'tempo':       round(float(row['tempo']), 1),
                'danceability':round(float(row['danceability']), 2),
            })
        return songs


# ── Test ──────────────────────────────────────────────────
if __name__ == "__main__":
    recommender = MusicRecommender('/mnt/user-data/uploads/spotify_track_dataset.zip')

    test_emotions = ['joy', 'sadness', 'anger', 'love', 'neutral']
    for emotion in test_emotions:
        print(f"\n{'='*50}")
        print(f"Emotion: {emotion.upper()}")
        print('='*50)
        songs = recommender.recommend(emotion, top_n=3)
        for i, song in enumerate(songs, 1):
            print(f"{i}. {song['track_name']} — {song['artists']}")
            print(f"   Genre: {song['genre']} | Popularity: {song['popularity']}")
            print(f"   Valence: {song['valence']} | Energy: {song['energy']}")
