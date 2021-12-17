import cv2
import mediapipe as mp
import numpy as np
from util import check_folder, get_filename, landmark_to_array
from os import path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = ['../data_science/dataset/Image/35/6_35_2_cam1.png']
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks = landmark_to_array(results)

        class_folder = path.normpath(file).split(path.sep)[-2]
        save_path = '../data_science/dataset/hand_landmarks/Image/' + class_folder + '/'
        check_folder(save_path)
        np.savetxt(save_path + get_filename(file) + '.csv', landmarks)

        print(landmarks)
