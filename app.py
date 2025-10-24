import os
import numpy as np
import librosa
import joblib
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import soundfile as sf
import io

# --- Initialize the Flask App and SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' # Change this to a random secret key
socketio = SocketIO(app)

# --- Load Trained Model and Tools ---
try:
    model = joblib.load("models/emotion_rf.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Model and tools loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    model = None

# This buffer will store incoming audio chunks
audio_buffer = bytearray()
BUFFER_SIZE_SECONDS = 2  # Process audio every 2 seconds
SAMPLE_RATE = 16000
BUFFER_SIZE_BYTES = BUFFER_SIZE_SECONDS * SAMPLE_RATE * 2 # 16-bit audio (2 bytes per sample)


# --- Feature Extraction Function ---
def extract_features(y, sr):
    """Extract MFCC, Chroma, and Spectral Contrast features from an audio signal."""
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        return np.hstack([mfcc, chroma, contrast])
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# --- Main Page Route ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Client connected."""
    print('✅ Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected."""
    global audio_buffer
    audio_buffer = bytearray() # Clear buffer on disconnect
    print('❌ Client disconnected')

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """Receives audio chunks from the client."""
    global audio_buffer
    
    # The incoming data is raw audio bytes. Append it to our buffer.
    audio_buffer.extend(data)

    # If we have enough data in the buffer, process it.
    if len(audio_buffer) >= BUFFER_SIZE_BYTES:
        
        # Get the chunk to process
        process_chunk = audio_buffer[:BUFFER_SIZE_BYTES]
        
        # Keep the rest for the next round
        audio_buffer = audio_buffer[BUFFER_SIZE_BYTES:]

        try:
            # Convert byte buffer to a numpy array of 16-bit integers
            y = np.frombuffer(process_chunk, dtype=np.int16)
            
            # Convert to float32, which is what librosa expects
            y = y.astype(np.float32) / 32767.0

            # 1. Extract features
            features = extract_features(y, sr=SAMPLE_RATE)
            if features is None:
                return

            # 2. Reshape and scale features
            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)

            # 3. Make prediction
            prediction = model.predict(features_scaled)
            emotion = label_encoder.inverse_transform(prediction)[0]
            
            # 4. Send the result back to the client
            print(f"🧠 Detected Emotion: {emotion}")
            emit('emotion_result', {'emotion': emotion})

        except Exception as e:
            print(f"An error occurred during processing: {e}")


if __name__ == '__main__':
    print("🚀 Starting real-time server...")
    # Use socketio.run for development. This will use eventlet if it's installed.
    socketio.run(app, debug=True, port=5000)