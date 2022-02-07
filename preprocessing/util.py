from os import walk, path, makedirs

import numpy as np
import pandas as pd

import mediapipe as mp
import cv2

######################################################################################################################
# File and folder operation FUNCTIONS
######################################################################################################################


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


######################################################################################################################
# Landmark extraction and calculation FUNCTIONS
######################################################################################################################


# Extract the landmark coordinates from the result object of hands.process and store them in a 21 x 3 array
def landmark_to_array(result):
    landmarks = np.zeros((21, 3))
    for i in range(21):
        landmarks[i, 0] = result.multi_hand_landmarks[-1].landmark[i].x
        landmarks[i, 1] = result.multi_hand_landmarks[-1].landmark[i].y
        landmarks[i, 2] = result.multi_hand_landmarks[-1].landmark[i].z

    return landmarks


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


# Return annotated image (to check plausibility) using the MediaPipe drawing utility
def annotate_image(image, result):
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


# Save the landmark coordinates in three csv files, one for each dimension
def save_1d(landmarks_3d, save_path):
    dimensions = ['x', 'y', 'z']
    for dim_i, dim in enumerate(dimensions):
        name = path.normpath(save_path).split(path.sep)[-1] + '_landmarks_' + dim + '.csv'
        np.savetxt(save_path + name, landmarks_3d[:, :, dim_i])


# Extract line of given file from DataFrame
def get_file_line(file_path, df):
    index = path.normpath(file_path).split(path.sep)[-2] + '/' + get_filename(file_path)

    return df.loc[index]


######################################################################################################################
# Coordinate system FUNCTIONS
######################################################################################################################


# Measure distance of different joints. Options for measure are:
# 'MCP': Index finger MCP to pinky MCP
# 'W-Index_MCP': Wrist to index finger MCP
# 'W-Pinky_MCP': Wrist to pinky MCP
# 'CMC': Wrist to thumb CMC
# 'aggregate': Use an aggregate measure of all above options with similar magnitude to MCP
def get_distance(coordinates, measure='MCP'):
    if measure == 'MCP':
        p0 = coordinates[:, 15:17]
        p1 = coordinates[:, 51:53]
    elif measure == 'W-Index_MCP':
        p0 = coordinates[:, 0:2]
        p1 = coordinates[:, 15:17]
    elif measure == 'W-Pinky_MCP':
        p0 = coordinates[:, 0:2]
        p1 = coordinates[:, 51:53]
    elif measure == 'CMC':
        p0 = coordinates[:, 0:2]
        p1 = coordinates[:, 3:5]
    elif measure == 'aggregate':
        return agg_distance(coordinates)

    d = (p0 - p1) ** 2
    d = np.sum(d, axis=1)

    return np.sqrt(d)


# Aggregate different distance measures. The output will be similar (but hopefully more robust) to MCP.
def agg_distance(coordinates):
    measures = ['MCP', 'W-Index_MCP', 'W-Pinky_MCP', 'CMC']
    weights = [0.25, 0.15, 0.2, 0.38]
    dist = 0

    for i, measure in enumerate(measures):
        d = get_distance(coordinates, measure)
        dist += d * weights[i]
    return dist


# Normalize coordinates to mitigate effect of hand-size. After normalization, the calculated distance will always
# equal the normalization_factor. The following distances (specified by measure) are available:
# 'MCP': Index finger MCP to pinky MCP
# 'W-Index_MCP': Wrist to index finger MCP
# 'W-Pinky_MCP': Wrist to pinky MCP
# 'CMC': Wrist to thumb CMC
# 'aggregate': Use an aggregate measure of all above options with similar magnitude to MCP
def normalize(coordinates, measure='MCP', normalization_factor=0.5):
    dist = get_distance(coordinates, measure)

    return normalization_factor * coordinates / dist.reshape((-1, 1))


# Shift the coordinate system origin to one of the nodes (0 is the wrist)
# If preserve relation is true, the coordinate system is flipped, such that points on the right of the node will be
# on the right in a cartesian coordinate system.
def shift_origin(coordinates, node=0, preserve_relation=False):
    new_origin = np.tile(coordinates[:, node:(node+3)], 21)
    coo_shift = coordinates.copy()
    coo_shift -= new_origin

    if preserve_relation:
        coo_shift *= -1

    return coo_shift


def flip_coordinates(coordinates, axis=0):
    """
    Flips (or mirrors) the coordinates of the specified axis by first multiplying with -1 and then adding 1.

    Parameters
    ----------
    coordinates : ndarray
        The 2d input array. Shape n x 63.
    axis : int
        The axis of the array that is flipped. 0 : x, 1 : y, 2 : z. The default is 0.

    Returns
    -------
    narray
        A copy of the input array flipped around axis. Shape n x 63.

    """
    coo_flip = coordinates.copy().reshape(-1, 21, 3)
    coo_flip[:, :, axis] = coo_flip[:, :, axis] * (-1) + 1

    return coo_flip.reshape(-1, 63)


######################################################################################################################
# Array utilities
######################################################################################################################


def find_min_max(input_arr):
    """
    Find and return the line-wise minima and maxima in an array.
    Parameters
    ----------
    input_arr : ndarray
        The 2d input array with n lines.
    Returns
    -------
    ndarray
        The minimum and maximum elements of the input. Shape n x 2.
    """
    min_max = np.amin(input_arr, axis=1).reshape(-1, 1)
    min_max = np.concatenate((min_max, np.amax(input_arr, axis=1).reshape(-1, 1)), axis=1)

    return min_max


def mode(input_arr):
    """
    Find the mode of an input, i.e., the most common value.

    Parameters
    ----------
    input_arr : array_like
        Input you want to find the mode of.

    Returns
    -------
    mode
        The smallest, most common value.

    """
    u_vals, counts = np.unique(input_arr, return_counts=True)
    index = np.argmax(counts)

    return u_vals[index]


######################################################################################################################
# Drawing and image FUNCTIONS
######################################################################################################################


def save_image(image, name, save_path):
    """
    Save image file as png.

    Parameters
    ----------
    image : ndarray
        The image to be saved.
    name : str
        The name of the saved image. Can be a file path, then the name is generated from the last element of that.
    save_path : str
        The path that the image is saved to. If it does not exist, it is created.

    Returns
    -------

    """
    check_folder(save_path)
    cv2.imwrite(save_path + '/' + get_filename(name) + '.png', cv2.flip(image, 1))


def img_coordinates(landmarks, img_height, img_width):
    """
    Calculate image coordinates from landmarks array.

    Parameters
    ----------
    landmarks : ndarray
        The landmarks to convert to image coordinates. Must be a 1d array with 63 elements.
    img_height : int
        The image height in pixels
    img_width : int
        The image width in pixels

    Returns
    -------
    landmarks_x : ndarray
        The x-image-coordinates. 1d array with 21 elements.
    landmarks_y : ndarray
        The y-image-coordinates. 1d array with 21 elements.

    """
    landmarks_x = landmarks[0::3]
    landmarks_y = landmarks[1::3]

    landmarks_x *= img_width
    landmarks_y *= img_height

    return landmarks_x, landmarks_y


def draw_landmarks(image, name, df):
    """
    Draw landmarks on image from DataFrame.

    Parameters
    ----------
    image : ndarray
        The image to draw the landmarks on.
    name : str
        (File) name of the image. Must be an index in df.
    df : DataFrame
        Contains the landmark coordinates to draw on the image.

    Returns
    -------
    ndarray
        The modified image including the landmarks.

    """
    landmarks = get_file_line(name, df).drop('Class')
    landmarks = landmarks.to_numpy()

    img = image
    img_height, img_width, _ = img.shape

    img_x, img_y = img_coordinates(landmarks, img_height, img_width)

    for i in range(img_x.shape[0]):
        cv2.drawMarker(img, (int(img_x[i]), int(img_y[i])), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=4)

    return img


def make_square(x_lims, y_lims):
    """
    Make an existing bounding box represented by x_lims and y_lims square such that the whole hand fits inside.

    Parameters
    ----------
    x_lims : ndarray
        The minimum and maximum x-coordinates in the coordinates of one hand. Shape n x 2.
    y_lims : ndarray
        The minimum and maximum y-coordinates in the coordinates of one hand. Shape n x 2.

    Returns
    -------
    x_lims : ndarray
        The modified x-limits such that x and y lengths are the same. Shape n x 2.
    y_lims : ndarray
        The modified x-limits such that x and y lengths are the same. Shape n x 2.

    """
    diff_dist = np.diff(x_lims) - np.diff(y_lims)

    for i, dist in enumerate(diff_dist):
        if dist < 0:
            x_lims[i, :] += np.array([[dist.item()/2, -dist.item()/2]]).reshape(-1)
        elif dist > 0:
            y_lims[i, :] += np.array([[-dist.item()/2, dist.item()/2]]).reshape(-1)

    return x_lims, y_lims
