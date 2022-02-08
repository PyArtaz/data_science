from os import walk, path, makedirs
import time

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
def annotate_image(image, result, jetson_nano_on):
    # Create necessary objects to use MediaPipe drawing utility
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    if(jetson_nano_on):

        for num, hand in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0),thickness=2 , circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,255,255),thickness=2 , circle_radius=2))
    else:
        mp_drawing_styles = mp.solutions.drawing_styles        

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
    new_origin = np.tile(coordinates[:, node:(node + 3)], 21)
    coo_shift = coordinates.copy()
    coo_shift -= new_origin

    if preserve_relation:
        coo_shift *= -1

    return coo_shift


# Create normalized coordinate system from bounding box
def normalize_by_bb(coordinates):
    x_lims, y_lims = create_bounding_box(coordinates, 320, 240, margin=0)
    coo_bb = coordinates.copy()

    coo_bb[:, 0:62:3] -= x_lims[:, 0].reshape((-1, 1))
    coo_bb[:, 0:62:3] /= (x_lims[:, 0] - x_lims[:, 1]).reshape((-1, 1))
    coo_bb[:, 0:62:3] += 0.5

    coo_bb[:, 1:62:3] -= y_lims[:, 1].reshape((-1, 1))
    coo_bb[:, 1:62:3] /= (y_lims[:, 0] - y_lims[:, 1]).reshape((-1, 1))

    return coo_bb


def coordinates_to_bb(coordinates, x_bound, y_bound):
    """
    Calculate the coordinates relative to a bounding box specified by x_bound and y_bound. The origin will be in the
    upper left corner of the bounding box, whereas the point (1, 1) will be in the lower left corner.
    Parameters
    ----------
    coordinates : ndarray
        The landmark coordinates of the hand in the image. Shape n x 63.
    x_bound : ndarray
        The minimum and maximum x-coordinates of the bounding box. Shape n x 2.
    y_bound : ndarray
        The minimum and maximum y-coordinates of the bounding box. Shape n x 2.
    Returns
    -------
    ndarray
        The shifted coordinates as described above. Same shape as coordinates (n x 63).
    """
    new_origin = np.zeros((x_bound.shape[0], 3))
    new_origin[:, 0] = x_bound[:, 0]
    new_origin[:, 1] = y_bound[:, 0]
    new_origin_tile = np.tile(new_origin, 21)
    coo_shift = coordinates.copy()
    coo_shift -= new_origin_tile

    x_range = np.diff(x_bound)
    y_range = np.diff(y_bound)
    z_range = (x_range + y_range) / 2

    coo_shift[:, 0:61:3] /= x_range.reshape((-1, 1))
    coo_shift[:, 1:62:3] /= y_range.reshape((-1, 1))
    coo_shift[:, 2:63:3] /= z_range.reshape((-1, 1))

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


def mode(input_arr, return_occurrences=False):
    """
    Find the mode of an input, i.e., the most common value.

    Parameters
    ----------
    input_arr : array_like
        Input you want to find the mode of.
    return_occurrences: bool
        If True, returns the number of occurrences. The default is False.

    Returns
    -------
    mode
        The smallest, most common value.

    """
    u_vals, counts = np.unique(input_arr, return_counts=True)
    try:
        index = np.argmax(counts)
        res = u_vals[index]
    except ValueError:
        return None

    if return_occurrences:
        return res, counts[index]
    else:
        return res


######################################################################################################################
# Drawing and image FUNCTIONS
######################################################################################################################


def calc_dps(fps_start):
    dps = 1 / (time.time() - fps_start)
    fps_start = time.time()

    return dps, fps_start


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


def create_bounding_box(landmarks, img_width, img_height, margin=10, square=False):
    """
    Calculate the relative coordinates of a bounding box around the hand. The same coordinate system as in the input
    landmarks is used.
    Parameters
    ----------
    landmarks : ndarray
        The landmark coordinates of the hand in the image. Shape n x 63.
    img_width : int, ndarray
        The image width in pixels.
    img_height : int, ndarray
        The image height in pixels.
    margin : int
        The amount of pixels that the bounding box would be larger on the image. The default is 10.
    square : bool
        Whether the bounding box is square or not. The bounding box will be square in pixels, unless
        img_width/img_height is 1. The default is False (rectangular).
    Returns
    -------
    x_bound : ndarray
        The minimum and maximum x-coordinates of the bounding box. Shape n x 2.
    y_bound : ndarray
        The minimum and maximum y-coordinates of the bounding box. Shape n x 2.
    """
    coordinates = landmarks.reshape(-1, 21, 3)
    img_width = np.asarray([img_width]) if np.isscalar(img_width) else np.asarray(img_width)
    img_height = np.asarray([img_height]) if np.isscalar(img_height) else np.asarray(img_height)

    x_bound = find_min_max(coordinates[:, :, 0])
    y_bound = find_min_max(coordinates[:, :, 1])

    x_margin = (margin / img_width).reshape(-1, 1)
    y_margin = (margin / img_height).reshape(-1, 1)

    x_bound += np.concatenate((-x_margin, x_margin), axis=1)
    y_bound += np.concatenate((-y_margin, y_margin), axis=1)

    if square:
        x_bound, y_bound = make_square_pic(x_bound, y_bound, img_width / img_height)

    return x_bound, y_bound


def draw_bounding_box(image, landmarks, fit_to_hand=True):
    """
    Draw a bounding box containing the hand on a given image.
    Parameters
    ----------
    image : ndarray
        The image to draw the bounding box on.
    landmarks : ndarray
        The landmark coordinates of the hand in the image. Must be a 1d array with 63 elements.
    fit_to_hand : bool
        Whether the bounding box fits the hand or has a predefined aspect ratio. If False, the box drawn on the image
        will be square. The default is True.
    Returns
    -------
    ndarray
        The modified image including the bounding box.
    """
    img = image
    img_height, img_width, _ = img.shape

    x_bounds, y_bounds = create_bounding_box(landmarks, img_width, img_height, square=not fit_to_hand)

    x_bounds = np.rint(x_bounds * img_width).astype(int)
    y_bounds = np.rint(y_bounds * img_height).astype(int)

    x_bounds, y_bounds = enforce_bounds(x_bounds, y_bounds, img_width, img_height)

    cv2.rectangle(img, (x_bounds[0, 0], y_bounds[0, 0]), (x_bounds[0, 1], y_bounds[0, 1]), (0, 0, 0), 2)

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
            x_lims[i, :] += np.array([[dist.item() / 2, -dist.item() / 2]]).reshape(-1)
        elif dist > 0:
            y_lims[i, :] += np.array([[-dist.item() / 2, dist.item() / 2]]).reshape(-1)

    return x_lims, y_lims


def make_square_pic(x_lims, y_lims, aspect_ratio):
    """
    Make an existing bounding box square in pixels such that the whole hand fits inside.
    The bounding box is represented by its border coordinates, x_lims and y_lims.
    Parameters
    ----------
    x_lims : ndarray
        The minimum and maximum x-coordinates in the coordinates of one hand. Shape n x 2.
    y_lims : ndarray
        The minimum and maximum y-coordinates in the coordinates of one hand. Shape n x 2.
    aspect_ratio : ndarray
        The aspect ratio of the input image(s), defined as width/height in pixels. Either 1 element or n elements.
        If an element is 1, the resulting bounding box is also square in coordinates.
    Returns
    -------
    x_lims : ndarray
        The modified x-limits such that x and y lengths are the same. Shape n x 2.
    y_lims : ndarray
        The modified x-limits such that x and y lengths are the same. Shape n x 2.
    """
    aspect_ratio = np.ones((x_lims.shape[0], 1)) * aspect_ratio.reshape(-1, 1)
    diff_dist = np.diff(x_lims) - np.diff(y_lims) / aspect_ratio

    for i, dist in enumerate(diff_dist):
        if dist < 0:
            x_lims[i, :] += np.array([[dist.item() / 2, -dist.item() / 2]]).reshape(-1)
        elif dist > 0:
            y_lims[i, :] += np.array([[-dist.item() * aspect_ratio[i] / 2, dist.item() * aspect_ratio[i] / 2]]).reshape(
                -1)

    return x_lims, y_lims


def enforce_bounds(x_border, y_border, img_width, img_height):
    """
    Ensure that the bounding box lies within in the image. If any coordinates don't, they will be transferred to the
    nearest point within the image.
    Parameters
    ----------
    x_border : ndarray
        The minimum and maximum x-coordinates in the coordinates of one hand. Shape n x 2.
    y_border : ndarray
        The minimum and maximum y-coordinates in the coordinates of one hand. Shape n x 2.
    img_width : int
        The image width in pixels.
    img_height: int
        The image height in pixels.
    Returns
    -------
    x_border : ndarray
        The modified x-limits such that all coordinates lie within the image. Shape n x 2.
    y_border : ndarray
        The modified x-limits such that all coordinates lie within the image. Shape n x 2.
    """
    if x_border[0, 0] < 0:
        x_border[0, 0] = 0
    if y_border[0, 0] < 0:
        y_border[0, 0] = 0
    if x_border[0, 1] > img_width:
        x_border[0, 1] = img_width
    if y_border[0, 1] > img_height:
        y_border[0, 1] = img_height

    return x_border, y_border


######################################################################################################################
# Prediction correction
######################################################################################################################


def prediction_checker(pred, correction_set):
    """
    Check, whether a passed prediction is often wrong and can be corrected by the specified correction_set.

    Parameters
    ----------
    pred : str
        The prediction made by the model to be corrected.
    correction_set: str
        The model that is used for correction. Options are 'Letters+Numbers' and 'ASLL'

    Returns
    -------
    bool
        True if it is a correctable but problematic prediction, False otherwise.

    """
    prob_inputs = asl_dict_min(correction_set)
    is_problematic = False

    if pred in prob_inputs:
        is_problematic = prob_inputs[pred]

    return is_problematic


def asl_dict_library(correction_set):
    """
    Return dictionaries containing characters that are often misunderstood by the asl+digits model.

    Parameters
    ----------
    correction_set : str
        The model that is used for correction. Options are 'Letters+Numbers' and 'ASLL'

    Returns
    -------
    dict
        A dictionary containing correctable, misunderstood characters.

    """
    if correction_set == 'Letters+Numbers':
        prob_inputs = {
            '1': True,
            '2': True,  # Not sure, has side effects
            '8': True,
            '9': True,
            'B': True,
            'G': False,  # Seems to be doing more harm
            'H': False,  # Same
            'R': True,
            'X': True,
            'Z': True
        }
    elif correction_set == 'ASLL':
        prob_inputs = {
            'A': False,  # Doesn't do much, supposed to help with T
            'B': True,
            'G': True,
            'H': True,
            'R': True,
            'S': False,  # Doesn't do much, supposed to help with T
            'X': True,
            'Z': True  # Necessary to help with A, D, M, N, but Z is difficult then
        }
    else:
        raise ValueError('Not a valid correction set!')

    return prob_inputs


def asl_dict_min(correction_set):
    """
    Return dictionaries containing characters that are often misunderstood by the asl+digits model.

    Parameters
    ----------
    correction_set : str
        The model that is used for correction. Options are 'Letters+Numbers' and 'ASLL'

    Returns
    -------
    dict
        A dictionary containing correctable, misunderstood characters.

    """
    if correction_set == 'Letters+Numbers':
        prob_inputs = {
            '9': True,
            'B': True,
            'G': False,  # Seems to be doing more harm
            'H': False,  # Same
            'R': True
        }
    elif correction_set == 'ASLL':
        prob_inputs = {
            'G': True,
            'H': True
        }
    else:
        raise ValueError('Not a valid correction set!')

    return prob_inputs
