from util import get_subfolders, get_files, check_folder, get_filename, landmark_to_array
import cv2
import mediapipe as mp
import numpy as np
from os import path
from time import time

dataset_path = '../data_science/dataset/Image/'
save_path = '../data_science/dataset/hand_landmarks/Image/'
class_folders = get_subfolders(dataset_path)
no_hand_detected = []
num_images = 0

start_time = time()

for folder in class_folders:
    IMAGE_FILES = get_files(folder)
    # Extract the class folder and create a complete file path for saving
    class_folder = path.normpath(folder).split(path.sep)[-1]
    save_path_complete = save_path + class_folder + '/'
    check_folder(save_path_complete)

    print('Scanning folder ' + class_folder)

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            num_images += 1
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                no_hand_detected.append(class_folder + '/' + get_filename(file))
                continue
            # Extract the landmark coordinates and store them in an array
            landmarks = landmark_to_array(results)
            # Save the coordinates in a csv file
            np.savetxt(save_path_complete + get_filename(file) + '.csv', landmarks)

print('No hand was detected in these files: ')
print(no_hand_detected)
print('No hands were detected in {} out of {} images.'.format(len(no_hand_detected), num_images))
print("\nRuntime", time()-start_time, "s")
