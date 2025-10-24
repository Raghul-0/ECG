import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

# Emotion mapping from RAVDESS file name codes
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(y, sr):
    """Extract MFCC, Chroma, and Spectral Contrast features from an audio signal."""
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    return np.hstack([mfcc, chroma, contrast])

def ensure_length(y, sr, duration=3):
    """Pad or trim audio to a fixed duration (in seconds)."""
    target_length = int(sr * duration)
    if len(y) < target_length:
        # pad with zeros (silence) if shorter
        y = np.pad(y, (0, target_length - len(y)))
    else:
        # trim if longer
        y = y[:target_length]
    return y

def build_feature_dataset(data_path="."):
    """Walk through all WAV files, extract features, and build a DataFrame."""
    features, labels = [], []
    for root, _, files in os.walk(data_path):
        for file in tqdm(files, desc=f"Processing {root}"):
            if not file.lower().endswith(".wav"):
                continue
            parts = file.split("-")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            emotion = EMOTION_MAP.get(emotion_code)
            if emotion is None:
                continue

            file_path = os.path.join(root, file)
            try:
                y, sr = sf.read(file_path)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)  # Convert to mono
                y = ensure_length(y, sr, duration=3)
                feats = extract_features(y, sr)
                features.append(feats)
                labels.append(emotion)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue

    df = pd.DataFrame(features)
    df['label'] = labels
    return df

if __name__ == "__main__":
    print("🔍 Extracting audio features from dataset...")
    df = build_feature_dataset(".")  # You can replace "." with a subfolder name like "ravdess"
    os.makedirs("features", exist_ok=True)
    df.to_csv("features/ravdess_features.csv", index=False)
    print(f"✅ Features saved to features/ravdess_features.csv with {len(df)} samples.")
