from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from app.app_modules import EmotionRecognitionAssistant
import shutil
import os

app = FastAPI(
    title="Emotion Recognition API",
    description="An API for analyzing emotions from various media types.",
    version="0.1.0",
)

# CORS middleware to allow requests from your React frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Make sure your React app's origin is here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = EmotionRecognitionAssistant()

@app.post("/analyze/", response_model=dict, summary="Analyze emotions from a file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyzes the emotions present in the provided file (text, audio, image, or video).
    Returns a dictionary containing the analysis results for each detected modality.
    """
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        analysis_result = assistant.analyze(file_path)
        os.remove(file_path)  # Clean up temporary file
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/", response_model=str, summary="Chat with the emotion analysis assistant")
async def chat_with_assistant(user_input: str):
    """
    Sends a message to the emotion analysis assistant and receives its response
    based on the previously analyzed file.
    """
    response = assistant.chat(user_input)
    return response

@app.get("/health/", status_code=200, summary="Health check")
async def health_check():
    """
    Returns a simple message indicating that the API is healthy.
    """
    return {"status": "healthy"}