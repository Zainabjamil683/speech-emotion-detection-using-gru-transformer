from flask import Flask, render_template, request
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
from pydub import AudioSegment

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model('best_gru_model.h5')
scaler = joblib.load('preprocessing_scaler.pkl')

# Emotion labels (same order used during training)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True, duration=3.0, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr)).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        y_harmonic = librosa.effects.harmonic(audio)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sr).T, axis=0)
        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['audio']
    if file.filename == '':
        return render_template('index.html', error='No selected file.')

    try:
        original_ext = file.filename.rsplit('.', 1)[-1].lower()
        temp_input_path = 'input_audio.' + original_ext
        temp_wav_path = 'converted_audio.wav'

        # Save uploaded file
        file.save(temp_input_path)

        # Convert to .wav if necessary
        if original_ext == 'mp3':
            audio = AudioSegment.from_mp3(temp_input_path)
            audio.export(temp_wav_path, format='wav')
        elif original_ext == 'wav':
            os.rename(temp_input_path, temp_wav_path)
        else:
            os.remove(temp_input_path)
            return render_template('index.html', error='Only .wav or .mp3 files are supported.')

        # Extract features
        features = extract_features(temp_wav_path)
        os.remove(temp_wav_path)
        if original_ext != 'wav':
            os.remove(temp_input_path)

        if features is None:
            return render_template('index.html', error='Could not extract features from audio.')

        features_scaled = scaler.transform([features])
        features_reshaped = np.expand_dims(features_scaled, axis=1)

        prediction = model.predict(features_reshaped)[0]
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]
        confidence = float(np.max(prediction))

        return render_template('index.html', emotion=predicted_emotion, confidence=round(confidence, 2))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
