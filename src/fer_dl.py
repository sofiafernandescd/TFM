'''
 # @ Author: Sofia Condesso
 # @ Create Time: 2025-03-03
 # @ Description: https://medium.com/@abhishekkhaiwale007/building-real-time-face-emotion-recognition-a-deep-dive-into-computer-vision-with-python-424b8806ea8c
 '''

import torch
import torch.nn as nn
import cv2
import numpy as np
import face_recognition
from torchvision import transforms
import torch.nn.functional as F
from collections import deque

# Define our emotion classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Dense layers
        self.fc1 = nn.Linear(128 * 12 * 12, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, len(EMOTIONS))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EmotionRecognizer:
    def __init__(self, model_path='emotion_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])


class EmotionBuffer:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
    
    def update(self, emotion):
        self.buffer.append(emotion)
        return self.get_dominant_emotion()
    
    def get_dominant_emotion(self):
        if not self.buffer:
            return None
        return max(set(self.buffer), key=self.buffer.count)
    
    def process_frame(self, frame):
        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        
        results = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            # Preprocess face
            tensor = self.transform(face_image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                emotion_idx = torch.argmax(output).item()
                emotion = EMOTIONS[emotion_idx]
                confidence = torch.softmax(output, dim=1)[0][emotion_idx].item()
                
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'location': face_location
            })
            
        return results
    
def process_frame1(self, frame, skip_frames=2):
    self.frame_count += 1
    if self.frame_count % skip_frames != 0:
        return self.last_results
    
    # Process frame as before
    results = self.detect_and_classify(frame)
    self.last_results = results
    return results

import matplotlib.pyplot as plt
def show_histogram(img, title):

    try:
        # if it's a colored image
        if img.shape[2] == 3:
            print("RGB IMAGE")
            color = ('blue', 'green', 'red')
            
            plt.figure(figsize=(12, 5))
            # fig, axs = plt.subplots(1, 3, figsize=(12, 5))

            # calculate histogram for each color
            for i, col in enumerate(color):
                # plot histogram
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                #axs[i].plot(histr, color=col)
                #axs[i].set_title(f'Histogram of {title} for color {col}')
            
            plt.title(f'Histogram of {title} for RGB')

            # show plot
            st.pyplot(plt)
            #plt.show()
            
            # transform to gray scale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        IndexError
                    
    plt.figure(figsize=(12, 5))
    # calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # plot histogram
    plt.plot(hist, color='black')
    plt.title(f'Histogram of {title} for gray scale')
    # show plot
    st.pyplot(plt)
    #plt.show()

    def display_image(img, title):

    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def blur_face(img):

    img_blur = img.copy()

    # haar cascade classifier to detect face patterns
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # detect faces
    faces = face_cascade.detectMultiScale(img_blur, 1.3, 5)

    # apply blurring
    for (x, y, w, h) in faces:
        img_blur[y:y+h, x:x+w] = cv2.blur(img_blur[y:y+h, x:x+w], (23, 23))

    return img_blur


def negative(img):

    # negative
    img = 255-img
    return img


def detect_eyes_mouth(img):

    img_eye_mouth = img.copy()

    # haar cascade face
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # haar cascade eyes
    eye_cascade = cv2.CascadeClassifier(
        # cv2.data.haarcascades + 'haarcascade_eye.xml')
        cv2.data.haarcascades +'haarcascade_eye_tree_eyeglasses.xml')
    # haar cascade mouth
    mouth_cascade = cv2.CascadeClassifier('models/haarcascade_mcs_mouth.xml')
        #cv2.data.haarcascades +'haarcascade_smile.xml')
    # detect faces
    faces = face_cascade.detectMultiScale(img_eye_mouth, 1.3, 5)
    # draw rectangles
    for (x, y, w, h) in faces:
        # face
        cv2.rectangle(img_eye_mouth, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # eyes
        roi = img_eye_mouth[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey),
                          (ex+ew, ey+eh), (0, 255, 0), 2)
        # mouth
        roi = img_eye_mouth[y:y+h, x:x+w]
        mouth = mouth_cascade.detectMultiScale(roi)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi, (mx, my),
                          (mx+mw, my+mh), (0, 0, 255), 2)

    return img_eye_mouth

def contrast(img):
    # adjust the contrast using image equalization
    if len(img.shape) == 3:         # color image
        channels = cv2.split(img)
        eq_channels = []
        for ch in channels:
            eq_channels.append(cv2.equalizeHist(ch))
        img_eq = cv2.merge(eq_channels)
    else:                           # gray scale image
        img_eq = cv2.equalizeHist(img)
    return img_eq


def automatic_brightness_and_contrast(image, alpha=1.0, beta=0.0):

    # automatic brightness and contrast adjustment
    img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return img



def jpeg_compression(img, quality):
    
    # encode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    # decode
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def png_compression(img, quality):
    
    # encode
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), quality]
    result, encimg = cv2.imencode('.png', img, encode_param)
    # decode
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def save_jpeg(img, file_name, quality):
    
    # jpeg compression
    decimg = jpeg_compression(img, quality)
    # save image
    cv2.imwrite(file_name, decimg)

def save_png(img, file_name, quality):
    
    # png compression
    decimg = png_compression(img, quality)
    # save image
    cv2.imwrite(file_name, decimg)

    
def main():
    recognizer = EmotionRecognizer()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = recognizer.process_frame(frame)
        
        # Draw results
        for result in results:
            top, right, bottom, left = result['location']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display emotion and confidence
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()