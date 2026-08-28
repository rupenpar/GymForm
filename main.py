import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
import subprocess
from src.pose_math import calculate_angle, calculate_midpt

if not os.path.exists("models/exercise_classifier.pkl"):
    print("Model Not found Training it rn!")
    subprocess.run(["python", "train.py"], check = True)

model = joblib.load("models/exercise_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


cap = cv.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        def get_pts(landmark_enum):
            return [lm[landmark_enum.value].x, lm[landmark_enum.value].y]

        r_shoulder = get_pts(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_shoulder = get_pts(mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_elbow = get_pts(mp_pose.PoseLandmark.RIGHT_ELBOW)
        l_elbow = get_pts(mp_pose.PoseLandmark.LEFT_ELBOW)
        r_wrist = get_pts(mp_pose.PoseLandmark.RIGHT_WRIST)
        l_wrist = get_pts(mp_pose.PoseLandmark.LEFT_WRIST)
        r_hip = get_pts(mp_pose.PoseLandmark.RIGHT_HIP)
        l_hip = get_pts(mp_pose.PoseLandmark.LEFT_HIP)
        r_knee = get_pts(mp_pose.PoseLandmark.RIGHT_KNEE)
        l_knee = get_pts(mp_pose.PoseLandmark.LEFT_KNEE)
        r_ankle = get_pts(mp_pose.PoseLandmark.RIGHT_ANKLE)
        l_ankle = get_pts(mp_pose.PoseLandmark.LEFT_ANKLE)
        
        mid_hip = calculate_midpt(r_hip, l_hip)

        input_data = pd.DataFrame({
            'right_elbow_right_shoulder_right_hip': [calculate_angle(r_elbow, r_shoulder, r_hip)],
            'left_elbow_left_shoulder_left_hip': [calculate_angle(l_elbow, l_shoulder, l_hip)],
            'right_knee_mid_hip_left_knee': [calculate_angle(r_knee, mid_hip, l_knee)],
            'right_hip_right_knee_right_ankle': [calculate_angle(r_hip, r_knee, r_ankle)],
            'left_hip_left_knee_left_ankle': [calculate_angle(l_hip, l_knee, l_ankle)],
            'right_wrist_right_elbow_right_shoulder': [calculate_angle(r_wrist, r_elbow, r_shoulder)],
            'left_wrist_left_elbow_left_shoulder': [calculate_angle(l_wrist, l_elbow, l_shoulder)]
        })

        prediction = model.predict(input_data)[0]
        exercise = label_encoder.inverse_transform([prediction])[0]
        
        # Display the predicted exercise on the video feed
        cv.putText(image, exercise.upper(), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('Gym Tracker', image)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

    
    







