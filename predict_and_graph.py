import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# === Load trained model and tools ===
model = joblib.load("models/emotion_rf.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# === Feature extraction ===
def extract_features(y, sr):
    """Extract MFCC, Chroma, and Spectral Contrast features from audio."""
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    return np.hstack([mfcc, chroma, contrast])

# === Predict emotions for a single clip ===
def predict_emotion_clip(path, window_s=1.5, hop_s=0.75):
    """Slide over the clip and predict emotion for each short window."""
    y, sr = librosa.load(path, sr=16000)
    window = int(window_s * sr)
    hop = int(hop_s * sr)

    times, preds = [], []
    for start in range(0, len(y) - window, hop):
        seg = y[start:start + window]
        feat = extract_features(seg, sr).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred = model.predict(feat_scaled)
        emotion = label_encoder.inverse_transform(pred)[0]
        times.append(start / sr)
        preds.append(emotion)

    return times, preds

# === Plot emotions over time ===
def plot_emotion_timeline(times, preds, title="Emotion Timeline"):
    """Plot emotions over time with color coding."""
    plt.figure(figsize=(12, 4))
    emotions = list(set(preds))
    colors = plt.cm.get_cmap("tab10", len(emotions))

    for i, emotion in enumerate(emotions):
        xs = [t for t, p in zip(times, preds) if p == emotion]
        ys = [emotion]*len(xs)
        plt.scatter(xs, ys, color=colors(i), label=emotion, s=30)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Emotion")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# === Batch process all audio files ===
def process_all_audio(folder_path="audio_speech_actors_01-24", output_csv="predictions.csv"):
    all_predictions = []

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
                times, preds = predict_emotion_clip(audio_path)
                for t, p in zip(times, preds):
                    all_predictions.append({
                        "file": file_name,
                        "actor": actor_folder,
                        "time_s": t,
                        "emotion": p
                    })

                plot_emotion_timeline(times, preds, title=f"{actor_folder} → {file_name}")

            except Exception as e:
                print(f"❌ Failed {audio_path}: {e}")

    # Save all predictions to CSV
    df = pd.DataFrame(all_predictions)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ All predictions saved to {output_csv}")

# === Main entry point ===
if __name__ == "__main__":
    process_all_audio()
