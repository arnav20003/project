import cv2
from deepface import DeepFace
import json
import random

# Load content recommendations from JSON file
with open('content_recommendations.json', 'r') as file:
    content_data = json.load(file)

def get_recommendation(emotion):
    music = random.choice(content_data['music'][emotion])
    joke = random.choice(content_data['jokes'][emotion])
    video = random.choice(content_data['videos'][emotion])
    return music, joke, video

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Face cascade loaded")

# Start capturing video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Detected faces: {len(faces)}")

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Analyze the face to predict emotions
            result = DeepFace.analyze(face_roi, actions=['emotion'])

            if len(result) > 0:
                emotion = result[0]['dominant_emotion']
                music, joke, video = get_recommendation(emotion)
                print(f"Detected Emotion: {emotion}")
                print(f"Music Recommendation: {music}")
                print(f"Joke Recommendation: {joke}")
                print(f"Video Recommendation: {video}")

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

    except Exception as e:
        print(f"Exception occurred: {e}")
        break

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
'''
import tkinter as tk
from tkinter import Label, Button, OptionMenu, StringVar, Frame
import cv2
from deepface import DeepFace
import json
import random
import threading
import time

class EmotionDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Emotion Detector")
        master.geometry("800x700")

        self.mode = StringVar(master)
        self.mode.set("Choose Mode")

        self.create_widgets()
        self.cap = None
        self.running = False
        self.previous_emotion = None
        self.last_recommendation_time = 0
        self.cooldown_period = 10  # Cooldown period in seconds

        # Load content recommendations from JSON file
        with open('content_recommendations.json', 'r') as file:
            self.content_data = json.load(file)
            print("Content Data Loaded: ", self.content_data)  # Debugging print statement

    def create_widgets(self):
        # Title Label
        self.label = Label(self.master, text="Welcome to the Emotion Detector", font=("Helvetica", 16))
        self.label.pack(pady=10)

        # Mode Selection
        self.mode_label = Label(self.master, text="Select Mode:", font=("Helvetica", 14))
        self.mode_label.pack(pady=5)

        self.mode_menu = OptionMenu(self.master, self.mode, "Continuous", "One-time")
        self.mode_menu.pack(pady=5)

        # Start and Stop Buttons
        self.start_button = Button(self.master, text="Start Detection", command=self.start_detection, font=("Helvetica", 12))
        self.start_button.pack(pady=10)

        self.stop_button = Button(self.master, text="Stop Detection", command=self.stop_detection, font=("Helvetica", 12))
        self.stop_button.pack(pady=10)

        # Video and Recommendation Frame
        self.video_frame = Frame(self.master, bg="black", width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)

        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.emotion_label = Label(self.master, text="Detected Emotion: None", font=("Helvetica", 14))
        self.emotion_label.pack(pady=10)

        self.recommendation_label = Label(self.master, text="", font=("Helvetica", 12), wraplength=500)
        self.recommendation_label.pack(pady=10)

    def start_detection(self):
        mode = self.mode.get()
        if mode == "Choose Mode":
            self.recommendation_label.config(text="Please select a mode before starting.")
            return

        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.previous_emotion = None
            threading.Thread(target=self.detect_emotion, args=(mode,)).start()

    def stop_detection(self):
        if self.running:
            self.running = False
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.video_label.config(image="")
            self.emotion_label.config(text="Detected Emotion: None")
            self.recommendation_label.config(text="")

    def get_recommendation(self, emotion):
        if emotion in self.content_data['music'] and emotion in self.content_data['jokes'] and emotion in self.content_data['videos']:
            music = random.choice(self.content_data['music'][emotion])
            joke = random.choice(self.content_data['jokes'][emotion])
            video = random.choice(self.content_data['videos'][emotion])
            return music, joke, video
        else:
            print(f"Emotion '{emotion}' not found in content data.")  # Debugging print statement
            return None, None, None

    def detect_emotion(self, mode):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.recommendation_label.config(text="Error: Could not read frame.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                result = DeepFace.analyze(face_roi, actions=['emotion'])

                if len(result) > 0:
                    emotion = result[0]['dominant_emotion']
                    current_time = time.time()
                    print(f"Detected emotion: {emotion}")  # Debugging print statement

                    music, joke, video = self.get_recommendation(emotion)
                    if music and joke and video:
                        if mode == "Continuous":
                            if (emotion != self.previous_emotion) or (current_time - self.last_recommendation_time > self.cooldown_period):
                                self.update_gui(emotion, music, joke, video)
                                self.previous_emotion = emotion
                                self.last_recommendation_time = current_time

                        elif mode == "One-time":
                            if emotion != self.previous_emotion:
                                self.update_gui(emotion, music, joke, video)
                                self.previous_emotion = emotion

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame, (640, 480))
            img = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
            self.video_label.config(image=img)
            self.video_label.image = img

            if mode == "One-time":
                break

        self.running = False

    def update_gui(self, emotion, music, joke, video):
        self.emotion_label.config(text=f"Detected Emotion: {emotion}")
        self.recommendation_label.config(
            text=f"Music: {music}\nJoke: {joke}\nVideo: {video}")

root = tk.Tk()
app = EmotionDetectorApp(root)
root.mainloop()
'''