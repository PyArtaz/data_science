from os import walk, path, makedirs
import numpy as np


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
