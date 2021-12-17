from os import walk, path, makedirs
import numpy as np


def get_subfolders(root_path):
    folders = next(walk(root_path), (None, [], None))[1]
    folders = [root_path + folder + '/' for folder in folders]

    return folders


def get_files(folder):
    files = next(walk(folder), (None, None, []))[2]
    files = [folder + file for file in files]

    return files


def check_folder(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)


def get_filename(filepath):
    return path.splitext(path.normpath(filepath).split(path.sep)[-1])[0]


def landmark_to_array(result):
    landmarks = np.zeros((21, 3))
    for i in range(21):
        landmarks[i, 0] = result.multi_hand_landmarks[0].landmark[i].x
        landmarks[i, 1] = result.multi_hand_landmarks[0].landmark[i].y
        landmarks[i, 2] = result.multi_hand_landmarks[0].landmark[i].z

    return landmarks
