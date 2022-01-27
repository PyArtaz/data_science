import sys
import os
import glob
import time

import cv2
import mediapipe as mp
import numpy as np
from util_mvp import landmark_to_array, flip_coordinates, mode, annotate_image
import bounding_box_mvp as bb
import pickle


def loadVid():
    model = load_latest_model()
    mp_hands = mp.solutions.hands

    # Specify whether video should be saved or not
    save_video = False
    window_name = "SigNum"

    # For webcam input:
    cap = cv2.VideoCapture(0)

    # FPS
    fps_start = 0

    # Smoothed prediction
    window_width = 20
    count = 0
    pred_list = [0] * window_width
    prob_list = np.ones((window_width, 38)) / window_width

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

            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            if results.multi_hand_landmarks:
                # Extract landmarks from image
                landmarks = landmark_to_array(results).reshape(1, -1)

                # Create bounding box and make it square
                box = bb.BoundingBox(landmarks, image.shape[1], image.shape[0])
                box.make_square()

                # Coordinate transformation
                landmarks_bb = box.coordinates_to_bb(landmarks)

                # Check for left hand and flip coordinates if left hand is detected
                hand = results.multi_handedness[-1].classification[0].label
                if hand == 'Left':
                    landmarks_bb = flip_coordinates(landmarks_bb)

                # Counting for rolling mode and mean calculations (Find window_width in line 90)
                if count == window_width:
                    count = 1
                else:
                    count += 1
                # Classify landmarks and use mode of the last window_width classes as prediction
                pred_list[count - 1] = model.predict(landmarks_bb)[0]
                prediction = mode(pred_list)

                # Calculate probabilities of each class and average the last window_with class probabilities
                prob_list[count - 1, :] = model.predict_proba(landmarks_bb)[0].reshape(1, -1)
                probability = np.mean(prob_list, axis=0)
                prob_prediction = model.classes_[np.argmax(probability)]
                probability = (probability * 100).round(2)
                probability_dict = dict(zip(model.classes_, probability))
                print(probability_dict)

                # Assemble the prediction string (1. with probability, 2. with prediction from mean probability)
                prediction_string = str(prediction) + '/' + str(prob_prediction) + '-' + str(
                    probability_dict.get(prediction))

                # Annotate the image with both, a bounding box and the landmark positions
                bb_image = box.draw(image)
                annotate_image(bb_image, results)

            else:
                prediction_string = ' '

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, prediction_string, (0, 430), font, 3, (0, 0, 255), 2)

            # FPS
            fps_end = time.time()
            time_diff = fps_end - fps_start
            fps = 1 / time_diff
            fps_start = fps_end

            fps_text = "FPS: {:.2f}".format(fps)
            cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

            if save_video:
                out.write(image)
            
            cv2.imshow(window_name,image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()


def load_latest_model():
    # Directory path containing saved models
    directory = ''  # dataset/saved_models/
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