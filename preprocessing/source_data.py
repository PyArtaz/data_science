from os import path
from time import time

import cv2
import mediapipe as mp
import numpy as np

from util import get_subfolders, check_folder, get_files, get_filename, landmark_to_array, annotate_image, save_image,\
    save_1d, gen_xyz_col_names, df_from_array
import bounding_box as bb

# Generate a database of hand landmarks in csv files from an image database. You have options to randomly save some
# annotated images and to save a separate landmark file for each image retaining the original folder structure.

# Specify relative file path of image database to be processed. Inside this folder, there are class folders with images.
dataset_path = '../data_science/dataset/Letters/'
# Specify file path of newly created hand landmarks database
save_path = '../data_science/dataset/hand_landmarks/Letters/'
# Optionally specify file path where annotated images should be saved
annotated_img_path = '../data_science/dataset/hand_landmarks/evaluation/Letters'
# Specify whether annotated images should be saved and with which probability
save_image_state = True
image_probability = 0.1
# Optionally save a landmark file for each image
save_separate = False

# Initializations
no_hand_detected = []
landmark_files = []
img_height = []
img_width = []
classes = []

class_folders = get_subfolders(dataset_path)
check_folder(save_path)
dataset_name = path.normpath(save_path).split(path.sep)[-1]
file_i = 0

rng = np.random.default_rng()
start_time = time()

for f_i, folder in enumerate(class_folders):
    IMAGE_FILES = get_files(folder)
    # Extract the class folder and create a complete file path for saving
    class_folder = path.normpath(folder).split(path.sep)[-1]
    if save_separate:
        save_path_complete = save_path + class_folder + '/'
        check_folder(save_path_complete)

    print('Scanning folder ' + class_folder)

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Check if a hand was detected, and if not, add file name to no_hand_detected and continue to next iteration
            if not results.multi_hand_landmarks:
                no_hand_detected.append(class_folder + '/' + get_filename(file))
                file_i += 1
                continue

            # Extract the landmark coordinates and store them in an array
            current_landmarks = landmark_to_array(results)[None, :, :]

            # Check for left hand and flip coordinates if left hand is detected
            hand = results.multi_handedness[-1].classification[0].label
            if hand == 'Left':
                current_landmarks[:, :, 0] = current_landmarks[:, :, 0] * (-1) + 1

            # Randomly save annotated images with image_probability to check annotation plausibility
            if save_image_state and rng.random() < image_probability:
                annotated_img = annotate_image(image, results)

                box = bb.BoundingBox(current_landmarks.reshape(1, -1), annotated_img.shape[1], annotated_img.shape[0])
                box.make_square()
                box.draw(annotated_img)

                save_image(annotated_img, file, annotated_img_path)

            # Save the landmark coordinates of each picture in a separate csv file
            if save_separate:
                np.savetxt(save_path_complete + get_filename(file) + '.csv', current_landmarks)
            else:
                if f_i == 0 and idx == file_i:
                    landmarks = current_landmarks.copy()
                else:
                    landmarks = np.concatenate((landmarks, current_landmarks))

                landmark_files.append(class_folder + '/' + get_filename(file))
                classes.append(class_folder)
                img_width.append(image.shape[1])
                img_height.append(image.shape[0])

if not save_separate:
    # Save landmark files, classes and landmarks separately, as well as files where no hands were detected (8 files)
    np.savetxt(save_path + dataset_name + '_landmark_files.csv', landmark_files, fmt='%s')
    np.savetxt(save_path + dataset_name + '_landmark_classes.csv', classes, fmt='%s')
    np.savetxt(save_path + dataset_name + '_no_hand_detected.csv', no_hand_detected, fmt='%s')
    np.savetxt(save_path + dataset_name + '_img_width.csv', img_width, fmt='%d')
    np.savetxt(save_path + dataset_name + '_img_height.csv', img_height, fmt='%d')
    save_1d(landmarks, save_path)

    # Construct and save a DataFrame containing the flattened landmarks, the landmark files and classes as index and
    # last column, respectively
    landmarks_df = df_from_array(landmarks, index=landmark_files, cols=gen_xyz_col_names(), classes=classes)
    landmarks_df.to_csv(save_path + dataset_name + '_landmarks.csv')

print('No hands were detected in {} out of {} images.'.format(len(no_hand_detected),
                                                              len(no_hand_detected) + len(landmark_files)))
print("\nRuntime", time() - start_time, "s")
