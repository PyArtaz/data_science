from os import walk, path, makedirs

import numpy as np
import pandas as pd

import mediapipe as mp
import cv2


# Return all the folders in a root directory as a list of strings including the relative path
def get_subfolders(root_path):
    folders = next(walk(root_path), (None, [], None))[1]
    folders = [root_path + folder + '/' for folder in folders]

    return folders


# Return the names of all files in a folder as a list of strings including the relative path
def get_files(folder):
    files = next(walk(folder), (None, None, []))[2]
    files = [folder + file for file in files]

    return files


# Check if a certain file path exists and if not, create it
def check_folder(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)


# Extract just the file name (without extension) from a file path
def get_filename(filepath):
    return path.splitext(path.normpath(filepath).split(path.sep)[-1])[0]


# Extract the landmark coordinates from the result object of hands.process and store them in a 21 x 3 array
def landmark_to_array(result):
    landmarks = np.zeros((21, 3))
    for i in range(21):
        landmarks[i, 0] = result.multi_hand_landmarks[0].landmark[i].x
        landmarks[i, 1] = result.multi_hand_landmarks[0].landmark[i].y
        landmarks[i, 2] = result.multi_hand_landmarks[0].landmark[i].z

    return landmarks


# Save the landmark coordinates in three csv files, one for each dimension
def save_1d(landmarks_3d, save_path):
    dimensions = ['x', 'y', 'z']
    for dim_i, dim in enumerate(dimensions):
        name = path.normpath(save_path).split(path.sep)[-1] + '_landmarks_' + dim + '.csv'
        np.savetxt(save_path + name, landmarks_3d[:, :, dim_i])


# Generates a list: [x0, y0, z0, ... , xi, yi, zi] where i is the features variable passed to the function
def gen_xyz_col_names(features=21):
    dir_names = ['x', 'y', 'z']
    col_names = []
    for feature in range(features):
        col_names += [dir_name + str(feature) for dir_name in dir_names]

    return col_names


# Create a pandas DataFrame from a 3d numpy array
def df_from_array(array, index, cols, classes):
    df = pd.DataFrame(data=np.reshape(array, (array.shape[0], -1)), index=index, columns=cols)
    df['Class'] = classes

    return df


# Return annotated image (to check plausibility)
def plot_random(image, result):
    # Create necessary objects to use MediaPipe drawing utility
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Draw landmarks in image
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return image


# Save image file to evaluation/dataset/ in hand_landmarks folder
def save_image(image, name, dataset):
    save_path = '../data_science/dataset/hand_landmarks/evaluation/' + dataset + '/'
    check_folder(save_path)
    cv2.imwrite(save_path + get_filename(name) + '.png', cv2.flip(image, 1))
