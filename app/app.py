import asyncio
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import time
import joblib

app = Flask(__name__)

# Maximum file size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024

# Load the pre-trained model
userModel = load_model("audio_type_classification.h5")

# Load the pre-trained model
moodModel = load_model("speaker_emotional_model.h5")

# Load the trained model
languageModel = joblib.load("language_model.pkl")

model_filename = "music_classification_model.joblib"
musicSpeechModel = joblib.load(model_filename)


# Function to extract audio features
def extract_music_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Calculate mean of each feature individually
    mfcc_mean = np.mean(mfcc) if mfcc.size > 0 else 0
    spectral_centroid_mean = np.mean(spectral_centroid) if spectral_centroid.size > 0 else 0
    spectral_bandwidth_mean = np.mean(spectral_bandwidth) if spectral_bandwidth.size > 0 else 0

    # Concatenate the means
    features = np.array([mfcc_mean, spectral_centroid_mean, spectral_bandwidth_mean])
    return features


# Function to extract MFCC features from an audio clip
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


def extract_languages(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)


def extract_features(audio_file, max_length=100):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Standardize feature length (pad or trim)
    if mfcc_features.shape[1] < max_length:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, max_length - mfcc_features.shape[1])), mode='constant')
    else:
        mfcc_features = mfcc_features[:, :max_length]
    return mfcc_features


# Asynchronous function to classify user type (AI or human)
async def classify_user_type(audio_file):
    # Extract features from test audio file
    test_features = extract_features(audio_file)

    # Reshape features to match the input shape of the model
    test_features = test_features.reshape(1, 20, 100)

    # Predict using the trained model
    prediction = userModel.predict(test_features)

    # Print the prediction
    if prediction >= 0.5:
        ai_percentage = int(prediction[0] * 100)
        human_percentage = int(100 - ai_percentage)
        predicted_class = 1
    else:
        human_percentage = int((1 - prediction[0]) * 100)
        ai_percentage = int(100 - human_percentage)
        predicted_class = 0

    # Map predicted class index to class label
    class_labels = ['human', 'AI']
    predicted_label = class_labels[predicted_class]
    result = {
        "userType": predicted_label,
        "human_percentage": human_percentage,
        "ai_percentage": ai_percentage
    }  # Dummy result
    return json.dumps(result)


# Asynchronous function to classify mood
async def classify_mood(audio_file):
    # Extract MFCC features
    mfcc_features = extract_mfcc(audio_file)
    # Reshape and normalize features
    mfcc_features = np.reshape(mfcc_features, (1, mfcc_features.shape[0], 1))
    # Make predictions
    predictions = moodModel.predict(mfcc_features)
    # Decode predictions
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprise', 'Sad', 'surprised']
    predicted_label = emotion_labels[np.argmax(predictions)]
    result = {"mood": predicted_label}  # Dummy result
    return json.dumps(result)


async def classify_music_speech(audio_file):
    # Extract features from the audio file
    features = extract_music_features(audio_file)
    # Reshape features for prediction
    features = features.reshape(1, -1)
    # Predict the class label
    predicted_label = musicSpeechModel.predict(features)[0]
    result = {"audioType": predicted_label}
    return json.dumps(result)


async def classify_language(audio_file):
    # Extract features from the test audio file
    test_features = extract_languages(audio_file)
    # Predict the language
    predicted_language = languageModel.predict([test_features])[0]
    result = {"language": predicted_language}
    return json.dumps(result)


def detect_noise_level(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Calculate the short-time Fourier transform (STFT)
    stft = librosa.stft(y)

    # Compute the power spectrogram
    power = np.abs(stft) ** 2

    # Calculate the mean power across frequency bins
    mean_power = np.mean(power, axis=0)

    # Normalize the mean power
    norm_mean_power = mean_power / np.max(mean_power)

    # Calculate the percentage of noise
    noise_percentage = np.sum(norm_mean_power > 0.2) / len(norm_mean_power) * 100  # Adjust threshold as needed

    # Categorize the noise level
    if noise_percentage > 60:
        noise_level = "High"
    elif noise_percentage > 30:
        noise_level = "Medium"
    else:
        noise_level = "Low"

    return noise_level


# Asynchronous function to process audio file and generate combined JSON
async def process_audio_async(audio_file):
    start_time = time.time()

    # Run both functions concurrently
    user_type_task = asyncio.create_task(classify_user_type(audio_file))
    mood_task = asyncio.create_task(classify_mood(audio_file))
    language_task = asyncio.create_task(classify_language(audio_file))
    music_speech_task = asyncio.create_task(classify_music_speech(audio_file))
   # noice_task = asyncio.create_task(detect_noise_level(audio_file))

    # Wait for both results to be ready
    user_type_result = await user_type_task
    mood_result = await mood_task
    language_result = await language_task
    music_speech_result = await music_speech_task
   # noice_result = await noice_task

    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    userobj = json.loads(user_type_result)
    moodobj = json.loads(mood_result)
    languageobj = json.loads(language_result)
    musicSpeechObj = json.loads(music_speech_result)

    result = {
        "status": "success",
        "analysis": {
            "detectedVoice": True,
            "voiceType": userobj.get("userType"),
            "confidenceScore": {
                "aiProbability": userobj.get("ai_percentage"),
                "humanProbability": userobj.get("human_percentage")
            },
            "additionalInfo": {
                "language": languageobj.get("language"),
                "emotionalTone": moodobj.get("mood"),
                "backgroundNoiseLevel": detect_noise_level(audio_file),
                "audioType": musicSpeechObj.get("audioType")
            }
        },
        "responseTime": response_time
    }
    return result


# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}


# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ping')
def health():
    return 'pong\n'

# API route for file upload and prediction
@app.route('/voice/analyse', methods=['POST'])
async def upload_file():
    # Check if the file is present in the reque
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the file to a temporary location
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_file_path)

        # Process the audio file asynchronously
        result = await process_audio_async(temp_file_path)

        # Remove the temporary file
        # os.remove(temp_file_path)

        return jsonify(result), 200

    return jsonify({"error": "Invalid file format"}), 400


if __name__ == '__main__':
    app.run(debug=True)
