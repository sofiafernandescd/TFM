"""This is the back-end app for the project Emotion Recognition in Multimedia Content.
Main requirements:
- receives a file and identify which modalities are in the file (text, audio, video etc...)
- acording to the files it uses a model to identify emotion in the existent modalities (one model for each modality)
- an LLM powered by ollama to generate a report with the results of the analysis
- service that works with the chat interaction with ollama model
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import mimetypes
import requests
from src.utils import load_model_with_weights
from ollama import Client as Ollama
from sklearn.svm import SVC
import librosa

from flask_restx import Api, Namespace, fields, Resource

# Initialize Flask application
app = Flask(__name__)
app.config['OLLAMA_URL'] = 'http://localhost:11434/api'

# Initialize Flask-RESTx Api
api = Api(app,
          version='1.0',
          title='Emotion API',
          description='Multimodal Emotion Analysis API',
          doc='/docs')

# Define namespaces
analysis_ns = Namespace('Analysis', description='File analysis operations')
chat_ns = Namespace('Chat', description='Chat interaction with the LLM')

# Add namespaces to the API
api.add_namespace(analysis_ns)
api.add_namespace(chat_ns)

# Model for Swagger documentation for Analysis
analysis_response_model = analysis_ns.model('AnalysisResult', {
    'modality': fields.String(description='Modality anlyzed'),
    'analysis': fields.String(description='Analysis results for the modality'), 
})

# Model for Swagger documentation for Chat
chat_request_model = chat_ns.model('ChatRequest', {
    'message': fields.String(required=True, description='User message'),
    'history': fields.List(fields.Raw, description='Chat history (list of {"role": "user/assistant", "content": "..."})')
})

chat_response_model = chat_ns.model('ChatResponse', {
    'response': fields.Raw(description='LLM response')
})

# Mock models (replace with actual implementations)
class EmotionModels:
    def __init__(self):
        self.audio_model = SVC()  # Replace with actual audio model
        self.text_model = SVC()  # Replace with actual text model
        self.video_model = SVC()  # Replace with actual video model

    def predict_audio(self, audio_path):
        # Load and preprocess audio data
        y, sr = librosa.load(audio_path)
        features = librosa.feature.mfcc(y=y, sr=sr)
        features = features.reshape(1, -1)  # Reshape for model input
        return self.audio_model.predict(features)

    def predict_text(self, text):
        # Preprocess text data
        return self.text_model.predict([text])

    def predict_video(self, video_path):
        # Load and preprocess video data
        return self.video_model.predict([video_path])  # Placeholder for video processing


# Initialize models
emotion_models = EmotionModels()
# Load models with weights
emotion_models.audio_model = load_model_with_weights("audio_model_path")
emotion_models.text_model = load_model_with_weights("text_model_path")
emotion_models.video_model = load_model_with_weights("video_model_path")
# Initialize Ollama client
ollama_client = Ollama()
# Set the model name
model_name = "tinyllama"  # Replace with your model name
# Set the URL for the Ollama API
ollama_url = app.config['OLLAMA_URL']
# Set the headers for the request
headers = {
    'Content-Type': 'application/json'
}
# Set the payload for the request
payload = {
    'model': model_name,
    'input': '',
    'temperature': 0.7,
    'max_tokens': 100
}
# Set the URL for the Ollama API

