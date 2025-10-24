import os
import numpy as np
import librosa
import joblib
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import soundfile as sf
import io
import tempfile # For handling uploaded files safely

# --- Initialize the Flask App and SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' # Change this to a random secret key
socketio = SocketIO(app)

# --- Load Trained Model and Tools ---
MODEL_LOADED = False
try:
    model = joblib.load("models/emotion_rf.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    MODEL_LOADED = True
    print("✅ Model and tools loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}. Prediction features will be disabled.")
    model = None
    label_encoder = None
    scaler = None
except Exception as e:
    print(f"❌ An unexpected error occurred loading models: {e}. Prediction features will be disabled.")
    model = None
    label_encoder = None
    scaler = None


# --- Configuration for Real-Time Processing ---
audio_buffer = bytearray()
BUFFER_SIZE_SECONDS = 2  # Process audio every N seconds
SAMPLE_RATE = 16000      # Sample rate model was trained on
BUFFER_SIZE_BYTES = BUFFER_SIZE_SECONDS * SAMPLE_RATE * 2 # 16-bit audio (2 bytes per sample)


# --- Feature Extraction Function ---
def extract_features(y, sr):
    """Extract MFCC, Chroma, and Spectral Contrast features from an audio signal."""
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        features = np.hstack([mfcc, chroma, contrast])
        return features
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# --- Main Page Route ---
@app.route('/')
def index():
    """Serves the main HTML page."""
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
    if not MODEL_LOADED:
        print("Model not loaded, skipping prediction.")
        emit('error', {'message': 'Model not loaded on server.'})
        return

    global audio_buffer
    audio_buffer.extend(data)

    # Process chunks of defined size
    while len(audio_buffer) >= BUFFER_SIZE_BYTES:
        process_chunk_bytes = audio_buffer[:BUFFER_SIZE_BYTES]
        audio_buffer = audio_buffer[BUFFER_SIZE_BYTES:] # Keep the remainder

        try:
            # Convert byte buffer to numpy array (16-bit PCM)
            y = np.frombuffer(process_chunk_bytes, dtype=np.int16)

            # Convert to float32 for librosa
            y_float = y.astype(np.float32) / 32767.0

            # 1. Extract features
            features = extract_features(y_float, sr=SAMPLE_RATE)
            if features is None:
                continue # Skip if feature extraction failed

            # Check for NaN or infinite values
            if not np.all(np.isfinite(features)):
                print("⚠️ Warning: Non-finite values detected in features. Skipping prediction for this chunk.")
                continue

            # 2. Reshape and scale features
            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)

            # 3. Make prediction
            prediction = model.predict(features_scaled)
            emotion = label_encoder.inverse_transform(prediction)[0]

            # 4. Send the result back to the client
            print(f"🧠 Real-time prediction: {emotion}")
            emit('emotion_result', {'emotion': emotion})

        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
            emit('error', {'message': f"Server processing error: {e}"})


# --- HTTP Route for File Upload Prediction ---
@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    """Handles audio file upload and returns the predicted emotion."""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded on server."}), 500

    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        try:
            # Save temporarily to read with librosa/soundfile
            with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(file.filename)[1]) as temp_audio:
                file.save(temp_audio.name)
                temp_audio.flush() # Ensure data is written

                # Load audio using librosa (handles various formats and resampling)
                y, sr = librosa.load(temp_audio.name, sr=SAMPLE_RATE) # Resample to model's expected rate

            # Ensure minimum length if needed (adjust if your model requires it)
            # min_length = int(BUFFER_SIZE_SECONDS * SAMPLE_RATE)
            # if len(y) < min_length:
            #     y = np.pad(y, (0, min_length - len(y)))

            if len(y) == 0:
                return jsonify({"error": "Audio file appears to be empty or corrupted."}), 400

            # 1. Extract features
            features = extract_features(y, sr)
            if features is None:
                 return jsonify({"error": "Feature extraction failed."}), 500

            # Check for NaN or infinite values
            if not np.all(np.isfinite(features)):
                print("⚠️ Warning: Non-finite values detected in uploaded file features.")
                return jsonify({"error": "Could not process audio features (NaN/Inf)."}), 500

            # 2. Reshape and scale features
            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)

            # 3. Make prediction
            prediction = model.predict(features_reshaped)
            emotion = label_encoder.inverse_transform(prediction)[0]

            print(f"📊 File prediction: {file.filename} -> {emotion}")
            return jsonify({"emotion": emotion})

        except Exception as e:
            print(f"❌ Error processing uploaded file {file.filename}: {e}")
            # Consider more specific error messages based on exception type
            if "Unsupported format" in str(e) or "Format not recognised" in str(e):
                 return jsonify({"error": f"Unsupported audio format: {os.path.splitext(file.filename)[1]}"}), 400
            else:
                return jsonify({"error": f"Server error during file processing: {e}"}), 500
    else:
        return jsonify({"error": "File processing failed."}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server...")
    # Use eventlet for WebSocket support in production environments
    # For local development, this automatically uses Flask's development server or eventlet if available
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
