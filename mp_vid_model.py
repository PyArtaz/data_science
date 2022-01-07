import sys
import os
import glob

import cv2
import mediapipe as mp
import numpy as np
from numpy.lib.function_base import median
from util_mvp import landmark_to_array, annotate_image
import pickle

        
#Defines with which probability the detected landmarks are printed to the console
print_probability = 0.0025
#Specify whether video should be saved or not
save_video = False

def loadVid():
        rng = np.random.default_rng()
        model = load_latest_model()
        mp_hands = mp.solutions.hands

        #For webcam input:
        cap = cv2.VideoCapture(0)
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

                #To improve performance, optionally mark the image as not writeable to pass by reference.
                top, bottom, left, right = 75, 275, 375, 575  # bottom_left, bottom_right, bottom_left+image_size, bottom_right+image_size
                roi = image[top:bottom, left:right]
                results = hands.process(roi)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                if results.multi_hand_landmarks:
                                annotate_image(roi, results)
                                # Extract landmarks from image (these could be passed to a classification algorithm)
                                landmarks = np.reshape(landmark_to_array(results), (1, -1))

                                # classify landmarks
                                prediction = model.predict(landmarks)

                                # calculate probabilities of each class
                                prediction_probabilities = model.predict_proba(landmarks)[0]

                                # format probabilities for display purposes
                                prediction_probabilities = list(np.around(np.array(prediction_probabilities * 100), 2))
                                prob_per_class_dictionary = dict(zip(model.classes_, prediction_probabilities))
                                print(prob_per_class_dictionary)

                                prediction_string = str(prediction[0]) + '-' + str(prob_per_class_dictionary.get(prediction[0]))
                else:
                                prediction_string = ' '
                cv2.rectangle(image, (right, top), (left, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, prediction_string, (0, 430), font, 3, (0, 0, 255), 2)
                if save_video:
                    out.write(image)
                # Flip the image horizontally for a selfie-view display.
                #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                cv2.imshow('MediaPipe Hands',image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        cap.release()


def load_latest_model():
        # Directory path containing saved models
        directory = ''      # dataset/saved_models/
        # necessary to load the latest saved model in the model folder
        list_of_files = glob.glob(directory + '*.pkl')  # '*' means all if need specific format then e.g.: '*.h5'
        if len(list_of_files) == 0:
            print("Could not find any model in directory:", directory)
            sys.exit()
        else:
            latest_file = max(list_of_files, key=os.path.getmtime)

            # load trained model
            with open(latest_file, 'rb') as f:
                model = pickle.load(f)

            return model
        
if __name__ == "__main__":
    loadVid()
    sys.exit() 