import os
import numpy as np
import librosa
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load trained model and tools ===
model = joblib.load("models/emotion_rf.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# === Feature extraction ===
def extract_features(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    return np.hstack([mfcc, chroma, contrast])

# === Predict emotions for a clip ===
def predict_emotion_clip(path, window_s=1.5, hop_s=0.75):
    y, sr = librosa.load(path, sr=16000)
    window = int(window_s * sr)
    hop = int(hop_s * sr)

    preds = []
    for start in range(0, len(y) - window, hop):
        seg = y[start:start + window]
        feat = extract_features(seg, sr).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred = model.predict(feat_scaled)
        emotion = label_encoder.inverse_transform(pred)[0]
        preds.append(emotion)
    return preds

# === Batch process all audio files and count emotions ===
def process_all_audio(folder_path="audio_speech_actors_01-24"):
    all_emotions = []

    for actor_folder in sorted(os.listdir(folder_path)):
        actor_path = os.path.join(folder_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file_name in sorted(os.listdir(actor_path)):
            if not file_name.endswith(".wav"):
                continue

            audio_path = os.path.join(actor_path, file_name)
            print(f"🔍 Processing: {audio_path}")
            try:
                preds = predict_emotion_clip(audio_path)
                all_emotions.extend(preds)
            except Exception as e:
                print(f"❌ Failed {audio_path}: {e}")

    # Count emotions
    emotion_counts = pd.Series(all_emotions).value_counts()

    # Plot histogram
    plt.figure(figsize=(8, 5))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="tab10")
    plt.title("Number of Segments per Emotion (All Audio)")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Segments")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print counts
    print("\n📊 Emotion segment counts:")
    print(emotion_counts)

if __name__ == "__main__":
    process_all_audio()
