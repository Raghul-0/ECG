# --- MUST BE FIRST ---
import eventlet
eventlet.monkey_patch()
# --- END MUST BE FIRST ---

import os
import numpy as np
import librosa
import joblib
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import soundfile as sf # soundfile might be needed by librosa under the hood
import io
import tempfile
import traceback

# --- Initialize the Flask App and SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_change_me!' # Use a better secret key
socketio = SocketIO(app, async_mode='eventlet') # Explicitly use eventlet

# --- Define Constants ---
BUFFER_SIZE_SECONDS = 2  # Process audio every N seconds
SAMPLE_RATE = 16000      # Sample rate model was trained on
BUFFER_SIZE_BYTES = BUFFER_SIZE_SECONDS * SAMPLE_RATE * 2 # 16-bit audio (2 bytes per sample)

# --- Load Trained Model and Tools ---
# Define global variables first, initialize to None
model = None
label_encoder = None
scaler = None
MODEL_LOADED = False

try:
    print("⏳ Loading machine learning models...")
    model = joblib.load("models/emotion_rf.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    MODEL_LOADED = True
    print("✅ Model and tools loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}. Prediction features will be disabled.")
except Exception as e:
    print(f"❌ An unexpected error occurred loading models: {e}. Prediction features will be disabled.")

# This buffer will store incoming audio chunks for real-time processing
audio_buffer = bytearray()


# --- Feature Extraction Function ---
def extract_features(y, sr):
    """Extract MFCC, Chroma, and Spectral Contrast features from an audio signal."""
    try:
        if y is None or len(y) < 2048: # Check for minimum length needed by librosa
             print("⚠️ Warning: Audio signal too short or empty for feature extraction.")
             return None

        # Calculate features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        features = np.hstack([mfcc, chroma, contrast])

        # Check for invalid numbers (NaN or infinity)
        if not np.all(np.isfinite(features)):
             print("⚠️ Warning: Non-finite values detected in extracted features.")
             return None

        return features
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        # traceback.print_exc() # Uncomment for more detailed debug logs if needed
        return None

# --- Main Page Route ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    print("Serving index.html")
    return render_template('index.html')

# --- WebSocket Event Handlers (Real-Time) ---
@socketio.on('connect')
def handle_connect():
    """Client connected."""
    print('✅ WebSocket Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected."""
    global audio_buffer
    audio_buffer = bytearray() # Clear buffer on disconnect
    print('❌ WebSocket Client disconnected')

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """Receives audio chunks from the client, buffers, and processes."""
    global audio_buffer, model, label_encoder, scaler, MODEL_LOADED, BUFFER_SIZE_BYTES, SAMPLE_RATE # Declare globals

    if not MODEL_LOADED:
        print("Model not loaded, skipping real-time prediction.")
        # Optionally emit an error, but might flood the client
        # emit('error', {'message': 'Model not loaded on server.'})
        return

    audio_buffer.extend(data)

    # Process chunks of defined size
    while len(audio_buffer) >= BUFFER_SIZE_BYTES:
        process_chunk_bytes = audio_buffer[:BUFFER_SIZE_BYTES]
        audio_buffer = audio_buffer[BUFFER_SIZE_BYTES:] # Keep the remainder

        try:
            # Convert byte buffer to numpy array (16-bit PCM) -> float32
            y = np.frombuffer(process_chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0

            features = extract_features(y, sr=SAMPLE_RATE)
            if features is None:
                print("Skipping audio chunk due to feature extraction issue.")
                continue # Skip this chunk if features couldn't be extracted

            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)

            prediction = model.predict(features_scaled)
            emotion = label_encoder.inverse_transform(prediction)[0]

            # print(f"🧠 Real-time prediction: {emotion}") # Can be noisy, uncomment if needed
            emit('emotion_result', {'emotion': emotion})

        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
            # traceback.print_exc() # Uncomment for detailed debug logs
            emit('error', {'message': f"Server processing error"})


# --- HTTP Route for File Upload Prediction ---
@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    """Handles audio file upload and returns the predicted emotion."""
    global model, label_encoder, scaler, MODEL_LOADED, SAMPLE_RATE # Declare globals

    print("Received request for /upload_predict")
    if not MODEL_LOADED:
        print("Model not loaded, cannot process file upload.")
        return jsonify({"error": "Model not loaded on server."}), 503 # Service Unavailable

    if 'audio_file' not in request.files:
        print("No 'audio_file' in request.files")
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['audio_file']

    if file.filename == '':
        print("No file selected.")
        return jsonify({"error": "No selected file."}), 400

    temp_path = None # Initialize temp_path
    try:
        if file:
            # Create a temporary file with the correct suffix
            suffix = os.path.splitext(file.filename)[1]
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = tf.name
            tf.close() # Close the file handle immediately after getting the name

            # Save the uploaded file content to the temporary path
            file.save(temp_path)
            print(f"⏳ Saved uploaded file temporarily to: {temp_path}")

            # Load audio using librosa
            print(f"⏳ Loading audio file: {file.filename} using librosa...")
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE) # Resample to model's expected rate
            print(f"✅ Audio loaded. Duration: {len(y)/sr:.2f}s, Sample Rate: {sr}")

            if len(y) == 0:
                 print(f"❌ Error: Audio file {file.filename} is empty or corrupted after loading.")
                 return jsonify({"error": "Audio file appears to be empty or corrupted."}), 400

            # 1. Extract features
            print(f"⚙️ Extracting features for {file.filename}...")
            features = extract_features(y, sr)
            if features is None:
                 print(f"❌ Error: Feature extraction failed for {file.filename}.")
                 return jsonify({"error": "Feature extraction failed. Audio might be too short or invalid."}), 400
            print("✅ Features extracted.")

            # 2. Reshape and scale features
            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)
            print("✅ Features scaled.")

            # 3. Make prediction
            print(f"🤖 Predicting emotion for {file.filename}...")
            prediction = model.predict(features_scaled)
            emotion = label_encoder.inverse_transform(prediction)[0]
            print(f"📊 File prediction successful: {file.filename} -> {emotion}")

            return jsonify({"emotion": emotion})
        else:
             print("❌ Error: File object not found in request.")
             return jsonify({"error": "File object is invalid."}), 400

    except librosa.PySoundFileError as e: # Catch specific soundfile/librosa errors
        print(f"❌ SoundFile/Librosa Error processing {file.filename}: {e}")
        return jsonify({"error": f"Error reading audio file format. Try WAV or MP3."}), 400
    except Exception as e:
        print(f"❌ Unhandled Exception in upload_predict for {file.filename}: {e}")
        traceback.print_exc() # Print full traceback to Render logs for debugging
        return jsonify({"error": f"An unexpected server error occurred during processing."}), 500
    finally:
        # Ensure temp file is always deleted if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"🗑️ Deleted temp file: {temp_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Error cleaning up temp file {temp_path}: {cleanup_error}")


# --- Start the Server ---
if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server...")
    # Use socketio.run with eventlet worker
    # Render uses the start command in its settings, so this part is mainly for local testing
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) # Keep debug=True for local testing
