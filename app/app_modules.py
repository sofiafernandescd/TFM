from litellm import completion
from deepface import DeepFace
import opensmile
import cv2
import pickle
import librosa
import numpy as np
import whisper
import os
import tempfile
from PIL import Image
from moviepy.editor import VideoFileClip


class FileProcessor:
    def __init__(self):
        self.file_types = {
            'text': ['txt', 'pdf', 'docx'],
            'audio': ['mp3', 'wav', 'sph'],
            'image': ['jpg', 'jpeg', 'png'],
            'video': ['mp4', 'avi', 'mov']
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = whisper.load_model("base") #sr.Recognizer()

    def process_file(self, file_path):
        """Main processing method"""
        if not os.path.isfile(file_path):
            return {"error": "File not found"}

        file_type = self._detect_file_type(file_path)

        try:
            if file_type == 'text':
                return self._process_text(file_path)
            elif file_type == 'audio':
                return self._process_audio(file_path)
            elif file_type == 'image':
                return self._process_image(file_path)
            elif file_type == 'video':
                return self._process_video(file_path)
            else:
                return {"error": "Unsupported file type"}
        except Exception as e:
            return {"error": str(e)}

    def _detect_file_type(self, file_path):
        """Detect file type category"""
        ext = file_path.split('.')[-1].lower()
        for category, extensions in self.file_types.items():
            if ext in extensions:
                return category
        return 'unknown'

    def _process_text(self, file_path):
        """Process text-based files"""
        ext = file_path.split('.')[-1].lower()
        text = ''

        if ext == 'txt':
            with open(file_path, 'r') as f:
                text = f.read()
        elif ext == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n'.join([page.extract_text() for page in reader.pages])
        elif ext == 'docx':
            doc = Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])

        return {"text": text}

    def _process_audio(self, file_path):
        """Process audio files with speech recognition"""
        audio, sr = librosa.load(file_path, sr=16000)
        transcript = self.recognizer.transcribe(audio)
        return {
            "text": transcript,
            "audio": {
                "raw": audio,
                "sample_rate": sr
            }
        }

    def _process_image(self, file_path):
        """Process image files with face detection"""
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return {"error": "No faces detected"}
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        return {"image": pil_image}

    def _process_video(self, file_path):
        """Process video files with frame extraction and audio processing"""
        result = {}
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpfile:
            video = VideoFileClip(file_path)
            if video.audio:
                video.audio.write_audiofile(tmpfile.name)
                audio_result = self._process_audio(tmpfile.name)
                result.update(audio_result)
            else:
                result["audio"] = None
                result["text"] = None

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # Every 1 second
        frame_count = 0
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    frames.append(pil_image)

            frame_count += 1

        cap.release()
        result["frames"] = frames
        return result

class TextEmotionRecognizer:
    def __init__(self, llm_model="phi4-mini"):
        self.llm_model = llm_model

    def analyze(self, text):
        try:
            response = completion(
                model=f"ollama_chat/{self.llm_model}",
                messages=[{
                    "content": f"Respond with only one word (lower case and no extra characters) from these emotions ['sad', 'happy', 'disgusted', 'surprised'] respecting to the most expressed emotion in the following piece of text: {text}",
                    "role": "user"}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return {"error": str(e)}

class SpeechEmotionRecognizer:
    def __init__(self, model_path='/Users/sofiafernandes/Documents/Repos/TFM/src/svm_model.pkl'):
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            self.model = None
            print(f"Warning: Speech emotion recognizer model not found at {model_path}")
        except Exception as e:
            self.model = None
            print(f"Error loading speech emotion recognizer: {e}")

    def extract_features(self, audio, sr):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_signal(audio, sr)
        return features.values

    def analyze(self, audio, sr):
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        features = self.extract_features(audio, sr)
        try:
            prediction = self.model.predict(features)
            return {"emotions": prediction.tolist()} # Return as list for JSON
        except Exception as e:
            return {"error": f"Error during audio emotion analysis: {e}"}

class FaceEmotionRecognizer:
    def __init__(self):
        self.backend = 'mtcnn'

    def analyze_image(self, image):
        try:
            image_array = np.array(image)
            results = DeepFace.analyze(
                img_path=image_array,
                actions=['emotion'],
                detector_backend=self.backend,
                enforce_detection=False,
                silent=True
            )
            if results:
                return {'emotions': results[0]['emotion'], 'dominant_emotion': results[0]['dominant_emotion']}
            else:
                return {'error': 'No face detected'}
        except Exception as e:
            return {'error': str(e)}

    def analyze_video_frames(self, frames):
        return [self.analyze_image(frame) for frame in frames]

class EmotionRecognitionAssistant:
    def __init__(self):
        self.file_processor = FileProcessor()
        self.text_recognizer = TextEmotionRecognizer()
        self.speech_recognizer = SpeechEmotionRecognizer()
        self.face_recognizer = FaceEmotionRecognizer()
        self.chatbot = ChatbotAssistant() # Use the assistant chatbot

    def analyze(self, file_path):
        processed_data = self.file_processor.process_file(file_path)

        if "error" in processed_data:
            return processed_data

        analysis_results = {}

        if "text" in processed_data and processed_data["text"]:
            analysis_results["text_emotion"] = self.text_recognizer.analyze(processed_data["text"])

        if "audio" in processed_data and processed_data["audio"] and processed_data["audio"]["raw"] is not None:
            analysis_results["audio_emotion"] = self.speech_recognizer.analyze(
                processed_data["audio"]["raw"], processed_data["audio"]["sample_rate"]
            )

        if "image" in processed_data and isinstance(processed_data["image"], Image.Image):
            analysis_results["face_emotion"] = self.face_recognizer.analyze_image(processed_data["image"])

        if "frames" in processed_data and processed_data["frames"]:
            analysis_results["face_emotion"] = self.face_recognizer.analyze_video_frames(processed_data["frames"])

        # Initialize chatbot with the analysis results
        self.chatbot.load_analysis(analysis_results)

        return analysis_results

    def chat(self, user_input):
        return self.chatbot.send_message(user_input)

class ChatbotAssistant:
    def __init__(self, llm_model="phi4-mini"):
        self.llm_model = llm_model
        self.analysis_summary = ""
        self.conversation_history = [{"role": "system", "content": "You are a helpful assistant that can discuss the results of an emotion analysis."}]

    def load_analysis(self, analysis_results):
        summary_parts = []
        if "text_emotion" in analysis_results and isinstance(analysis_results["text_emotion"], str):
            summary_parts.append(f"The detected emotion in the text was: {analysis_results['text_emotion']}.")
        elif "text_emotion" in analysis_results and "error" in analysis_results["text_emotion"]:
            summary_parts.append(f"There was an error analyzing the text: {analysis_results['text_emotion']['error']}.")

        if "audio_emotion" in analysis_results and isinstance(analysis_results["audio_emotion"], dict) and "emotions" in analysis_results["audio_emotion"]:
            summary_parts.append(f"The detected emotions in the audio were: {analysis_results['audio_emotion']['emotions']}.")
        elif "audio_emotion" in analysis_results and "error" in analysis_results["audio_emotion"]:
            summary_parts.append(f"There was an error analyzing the audio: {analysis_results['audio_emotion']['error']}.")

        if "face_emotion" in analysis_results:
            if isinstance(analysis_results["face_emotion"], dict) and "dominant_emotion" in analysis_results["face_emotion"]:
                summary_parts.append(f"The dominant facial emotion was: {analysis_results['face_emotion']['dominant_emotion']}.")
            elif isinstance(analysis_results["face_emotion"], list):
                dominant_emotions = [res.get('dominant_emotion') for res in analysis_results["face_emotion"] if isinstance(res, dict) and 'dominant_emotion' in res]
                if dominant_emotions:
                    summary_str = ", ".join(dominant_emotions)
                    summary_parts.append(f"The dominant facial emotions across frames were: {summary_str}.")
                elif analysis_results["face_emotion"] and all("error" in res for res in analysis_results["face_emotion"]):
                    summary_parts.append(f"There were errors analyzing faces in the frames.")
            elif isinstance(analysis_results["face_emotion"], dict) and "error" in analysis_results["face_emotion"]:
                summary_parts.append(f"There was an error analyzing the face: {analysis_results['face_emotion']['error']}.")

        if summary_parts:
            self.analysis_summary = "Here's a summary of the emotion analysis: " + " ".join(summary_parts)
            self.conversation_history.append({"role": "system", "content": self.analysis_summary})
        else:
            self.conversation_history.append({"role": "system", "content": "No emotions were detected or there were errors during analysis."})

    def send_message(self, message):
        self.conversation_history.append({"role": "user", "content": message})
        try:
            response = completion(
                model=f"ollama_chat/{self.llm_model}",
                messages=self.conversation_history,
                stream=False
            )
            bot_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            return bot_response
        except Exception as e:
            return f"Error generating chatbot response: {str(e)}"

if __name__ == "__main__":
    assistant = EmotionRecognitionAssistant()
    file_path = '/Users/sofiafernandes/Documents/Repos/MEIM-ano1-sem2/TFM-SC/01-01-05-02-02-01-01.mp4' # Replace with a valid file path
    analysis_result = assistant.analyze(file_path)

    print("Analysis Results:")
    print(analysis_result)

    if assistant.chatbot.analysis_summary:
        print("\nChat with the Assistant:")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = assistant.chat(user_input)
            print(f"Assistant: {response}")
    else:
        print("\nNo analysis summary available to chat about.")