from os import path

import numpy as np
import pandas as pd

from util import flip_coordinates, gen_xyz_col_names
import bounding_box as bb

# Normalize the hand landmark coordinates created by source_data using a square bounding box and save them to a new file

# Specify relative file path of hand landmarks database created by source_data
data_path = '../data_science/dataset/hand_landmarks/Letters/'
# Specify file path of the normalized hand landmarks file
save_path = '../data_science/dataset/hand_landmarks/Letters/'

# Specify whether all images have the same size
# Image width and height are loaded from files if dynamic_image_format is True
dynamic_image_format = False
image_width, image_height = 640, 480

dataset_name = path.normpath(data_path).split(path.sep)[-1]
input_file = data_path + dataset_name + '_landmarks.csv'
output_file = save_path + dataset_name + '_landmarks_bb_squarePix_flip.csv'

if dynamic_image_format:
    image_width = np.loadtxt(data_path + dataset_name + '_img_width.csv')
    image_height = np.loadtxt(data_path + dataset_name + '_img_height.csv')

# Construct DataFrame from csv file specified by data and convert numpy array without class column
df = pd.read_csv(input_file, header=0, index_col=0)
arr = df.to_numpy()[:, 0:63]

# Calculate the bounding box coordinates for each line
box = bb.BoundingBox(df.to_numpy()[:, 0:63], img_width=image_width, img_height=image_height, margin=10)
box.make_square()
# Calculate the landmark coordinates relative to the bounding box specified above
arr_bb = box.coordinates_to_bb(arr)
# Flip the x-coordinates (to pretend there are also left hands in the data)
arr_bb = flip_coordinates(arr_bb)

# Construct and save a DataFrame containing the landmarks relative to the bounding box, the landmark files and
# classes as index and last column, respectively
df_bb = pd.DataFrame(data=arr_bb, index=df.index, columns=gen_xyz_col_names())
df_bb['Class'] = df['Class']
df_bb.to_csv(output_file)
