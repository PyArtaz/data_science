import cv2
import mediapipe as mp
import numpy as np
from util_mvp import landmark_to_array, annotate_image

# Defines with which probability the detected landmarks are printed to the console
print_probability = 0.0025
# Specify whether video should be saved or not
save_video = False

rng = np.random.default_rng()
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(1)
if save_video:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (frame_width, frame_height))

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            annotate_image(image, results)
            # Extract landmarks from image (these could be passed to a classification algorithm)
            landmarks = np.reshape(landmark_to_array(results), (1, -1))
            if rng.random() < print_probability:
                print(landmarks)

        if save_video:
            out.write(image)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
