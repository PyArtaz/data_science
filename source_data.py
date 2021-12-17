from util import get_subfolders, get_files, check_folder, get_filename, landmark_to_array, save_1d
import cv2
import mediapipe as mp
import numpy as np
from os import path
from time import time

dataset_path = '../data_science/dataset/Image/'
save_path = '../data_science/dataset/hand_landmarks/Image/'
class_folders = get_subfolders(dataset_path)
no_hand_detected = []
landmark_files = []
classes = []

file_i = 0
save_separate = False

start_time = time()

for f_i, folder in enumerate(class_folders):
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
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                no_hand_detected.append(class_folder + '/' + get_filename(file))
                file_i += 1
                continue
            # Extract the landmark coordinates and store them in an array
            if save_separate:
                landmarks = landmark_to_array(results)[None, :, :]
                # Save the landmark coordinates of each picture in a separate csv file
                np.savetxt(save_path_complete + get_filename(file) + '.csv', landmarks)
            else:
                if f_i == 0 and idx == file_i:
                    landmarks = landmark_to_array(results)[None, :, :]
                else:
                    landmarks = np.concatenate((landmarks, landmark_to_array(results)[None, :, :]))

                landmark_files.append(class_folder + '/' + get_filename(file))
                classes.append(class_folder)

if not save_separate:
    save_1d(landmarks, save_path)
    dom_name = path.normpath(save_path).split(path.sep)[-1]
    np.savetxt(save_path + dom_name + '_landmark_files.csv', landmark_files, fmt='%s')
    np.savetxt(save_path + dom_name + '_landmark_classes.csv', classes, fmt='%s')


print('No hand was detected in these files: ')
print(no_hand_detected)
print('No hands were detected in {} out of {} images.'.format(len(no_hand_detected), len(landmark_files)))
print("\nRuntime", time()-start_time, "s")
