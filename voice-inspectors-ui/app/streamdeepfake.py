import streamlit as st
import os
import plotly.express as px
import json
import os
import time
import joblib
import asyncio
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pyaudio
import wave
import random
import speech_recognition as sr
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import sys

# Load the pre-trained model
userModel = load_model("app/audio_type_classification.h5")

# Load the pre-trained model
moodModel = load_model("app/speaker_emotional_model.h5")

# Load the trained model
languageModel = joblib.load("app/language_model.pkl")

model_filename = "app/music_classification_model.joblib"
musicSpeechModel = joblib.load(model_filename)

model_filename = "app/live_vs_recorded_model.joblib"
classifier = joblib.load(model_filename)

st.set_page_config(layout="wide")

st.title("Voice Detectives")

col1, col2 = st.columns([2, 1])  # Adjust the widths as needed


def get_text(audio_file):
    try:
        # Load the audio file
        if audio_file.endswith('.wav'):
            audio = AudioSegment.from_wav(audio_file)
        elif audio_file.endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_file)
        else:
            print("Unsupported audio format")
            musicType = 'Music'

        # Set the chunk size (in milliseconds)
        chunk_size_ms = 1000  # 1 second

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Transcribe speech from each chunk
        transcription = ""
        for i in range(0, len(audio), chunk_size_ms):
            if i > 10000:
                musicType = 'Music'
                result = {"musicType": musicType}
                return json.dumps(result)
            chunk = audio[i:i + chunk_size_ms]
            print(f"i === {i}")

            # Export the chunk as a temporary WAV file
            chunk.export("temp.wav", format="wav")

            # Recognize speech from the chunk
            with sr.AudioFile("temp.wav") as source:
                audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data)
                transcription += text + " "
                if sys.getsizeof(transcription) > 1:
                    print(f"{transcription}")
                    musicType = 'Speech'
                    result = {"musicType": musicType}
                    return json.dumps(result)
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Error:", e)

    except CouldntDecodeError:
        print("Error: Could not decode audio file")

    musicType = 'Music'
    result = {"musicType": musicType}
    return json.dumps(result)


def extract_live_features(audio_file, mfcc=True, chroma=True, mel=True):
    y, sample_rate = librosa.load(audio_file)
    features = []
    if chroma:
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sample_rate)
        features.append(chroma_stft.T)
    if mel:
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
        features.append(mel_spectrogram.T)
    if mfcc:
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate)
        features.append(mfccs.T)
    return np.hstack(features).reshape(1, -1)  # Reshape features for prediction


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


def detect_audio_content(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Compute the short-time Fourier transform (STFT)
    stft = librosa.stft(y)

    # Calculate the mean energy of the STFT across frequency bins
    energy = np.mean(np.abs(stft), axis=0)

    # Threshold for determining if speech is present
    speech_threshold = np.mean(energy) * 5  # Adjust this threshold as needed

    # Detect speech based on energy threshold
    if np.max(energy) > speech_threshold:
        return "Speech"
    else:
        return "Music"


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
def classify_user_type(audio_file):
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
def classify_mood(audio_file):
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


def classify_music_speech(audio_file):
    # Extract features from the audio file
    features = extract_music_features(audio_file)
    # Reshape features for prediction
    features = features.reshape(1, -1)
    # Predict the class label
    predicted_label = musicSpeechModel.predict(features)[0]
    result = {"audioType": predicted_label}
    return json.dumps(result)


def classify_language(audio_file):
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
def process_audio_async(audio_file):
    start_time = time.time()

    user_type_task = classify_user_type(audio_file)
    mood_task = classify_mood(audio_file)
    language_task = classify_language(audio_file)
    music_speech_task = get_text(audio_file)
    # music_speech_task = classify_music_speech(audio_file)

    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    userobj = json.loads(user_type_task)
    moodobj = json.loads(mood_task)
    languageobj = json.loads(language_task)
    musicSpeechObj = json.loads(music_speech_task)

    musicType = musicSpeechObj.get("musicType")

    if musicType == 'Music':
        result = {
            "status": "success",
            "analysis": {
                "detectedVoice": True,
                "voiceType": 'Music',
                "confidenceScore": None,
                "additionalInfo": {
                    "language": None,
                    "emotionalTone": None,
                    "backgroundNoiseLevel": detect_noise_level(audio_file),
                    "audioType": 'Music'
                }
            },
            "responseTime": response_time
        }
    else:
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
                    "audioType": 'Speech'
                }
            },
            "responseTime": response_time
        }
    return result


def record_audio(file_path, duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    col2.write("Recording...")
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    col2.empty()
    col2.write("Recording complete!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def display_analysis_results(file_path):
    column1, column2, column3 = st.columns(3)
    with column1:
        st.info("Your results are below")
        result = process_audio_async(file_path)
        st.json(result)
    with column2:
        st.info("Your uploaded audio is below")
        st.audio(file_path)
        # Load the audio file
        audio_clip, _ = librosa.load(file_path, sr=None)

        # Create a time array
        time = librosa.times_like(audio_clip)

        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=audio_clip, mode='lines', name='Audio Waveform'))

        # Customize the layout
        fig.update_layout(
            title="Waveform Plot",
            xaxis_title="Time",
            yaxis_title="Amplitude"
        )

        # Plot the figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with column3:
        st.info("Disclaimer")
        st.warning("These classification or detection mechanisms are not always accurate. They should be "
                   "considered as a strong signal and not the ultimate decision makers.")


def main():
    uploaded_file = col1.file_uploader("Upload an audio file", type=["mp3", "wav"])

    folder_path = 'uploads'  # Change this to your desired folder path

    os.makedirs(folder_path, exist_ok=True)

    file_path = ''
    record_path = ''

    col2.write("Record an audio file")
    if col2.button("Record Audio"):
        random_number = random.randint(1, 100)
        file_name = str(random_number) + "recorded_audio.wav"
        file_path = os.path.join(folder_path, file_name)
        record_path = os.path.join(folder_path, file_name)
        record_audio(record_path)
        features = extract_live_features(file_path)
        # Make prediction using the trained model
        prediction = classifier.predict(features)
        predicted_label = prediction[0]
        col2.success(f"Liveliness :  {predicted_label}")
        display_analysis_results(file_path)

    # if col2.button('Record'):
    #     file_path = os.path.join(folder_path, 'recorded_audio.wav')
    #     record_path = os.path.join(folder_path, 'recorded_audio.wav')
    #     record_audio(record_path)
    #     display_analysis_results(file_path)

    if uploaded_file is not None:
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # st.write(f"file path : {file_path}")

    if uploaded_file is not None:
        if st.button("Analyse Audio"):
            display_analysis_results(file_path)


if __name__ == '__main__':
    main()
