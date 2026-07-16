import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
import os
import sys

# -------- TTS Engine --------
engine = pyttsx3.init()
spoken_faces = set()

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

HAAR_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
AGE_PROTOTXT = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTOTXT = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")
EMOTION_MODEL = os.path.join(MODELS_DIR, "emotion_model.h5")

# Labels
AGE_LIST = ['(0-6)', '(7-12)', '(13-19)', '(20-30)', '(31-45)', '(46-60)', '(61+)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------- Load Models --------
def load_models():
    # Face detection
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    if face_cascade.empty():
        print("Error: Could not load Haar cascade xml!")
        sys.exit(1)

    # Age & Gender DNN
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_MODEL)

    # Emotion Keras model
    emotion_model = load_model(EMOTION_MODEL)

    return face_cascade, age_net, gender_net, emotion_model

# -------- Face Detection --------
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    return faces

# -------- Predictions --------
def predict_age_gender(face_img, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227),
                                 (78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False)
    # Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return age, gender

def predict_emotion(face_img, emotion_model):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48,48))
    face_resized = np.expand_dims(face_resized, axis=[0,-1])
    face_resized = face_resized / 255.0
    preds = emotion_model.predict(face_resized)
    emotion = EMOTION_LIST[np.argmax(preds)]
    return emotion

# -------- Speak --------
def speak_prediction(age, gender, emotion):
    text = f"You look {emotion.lower()}, around {age} years old, and {gender.lower()}."
    engine.say(text)
    engine.runAndWait()

# -------- Main Loop --------
def main():
    face_cascade, age_net, gender_net, emotion_model = load_models()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, face_cascade)

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]

            age, gender = predict_age_gender(face, age_net, gender_net)
            emotion = predict_emotion(face, emotion_model)

            label = f"{gender}, {age}, {emotion}"
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            face_center = (x + w//2, y + h//2)
            if face_center not in spoken_faces:
                speak_prediction(age, gender, emotion)
                spoken_faces.add(face_center)

        cv2.imshow("Face Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.stop()

if __name__ == "__main__":
    main()
