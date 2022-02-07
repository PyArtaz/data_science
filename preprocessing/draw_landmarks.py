import cv2
import pandas as pd

from util import draw_landmarks, save_image, get_file_line, normalize, shift_origin, flip_coordinates
import bounding_box as bb

# Demonstration script to showcase different normalization options and the bounding box functionality.

# Specify relative file path of hand landmarks database file created by source_data ..._landmarks.csv
data = '../data_science/dataset/hand_landmarks/Image/Image_landmarks.csv'
# Specify file path of the image to be annotated. An entry must be present in the database file above.
img_path = '../data_science/dataset/Image/1/3_1_2_cam1.png'
# Specify file path where the annotated image should be saved
annotated_img_path = '../data_science/dataset/hand_landmarks/evaluation/Image/post'

# Construct DataFrame from csv file specified by data and extract line specified by img_path without class label
df = pd.read_csv(data, header=0, index_col=0)
line = get_file_line(img_path, df).to_numpy().reshape((1, -1))[:, 0:63]

# Normalize the coordinates (works for n x 63 numpy arrays)
line_n = normalize(line, measure='MCP', normalization_factor=0.5)
df_n = normalize(df.to_numpy()[:, 0:63])

# Shift the origin to the wrist position (works for n x 63 numpy arrays). Could also shift the normalized coordinates.
line_s = shift_origin(line)
df_ns = shift_origin(df_n)

# Flip the x-coordinates (to pretend there are also left hands in the data)
line_flip = flip_coordinates(line)
df_flip = flip_coordinates(df.to_numpy()[:, 0:63])

# Create bounding boxes and make them square
box_df = bb.BoundingBox(df.to_numpy()[:, 0:63], 320, 240)
box_ll = bb.BoundingBox(line, 320, 240)
box_df.make_square()
box_ll.make_square()

# Calculate the landmark coordinates relative to the bounding box specified above
df_bb = box_df.coordinates_to_bb(df.to_numpy()[:, 0:63])
line_bb = box_ll.coordinates_to_bb(line)

# Load the image specified by img_path and annotate it with a bounding box and the landmark positions
image = cv2.flip(cv2.imread(img_path), 1)
annotated_image = box_ll.draw(image)
annotated_image = draw_landmarks(annotated_image, img_path, df)

# Save the annotated image to the annotated_img_path using the name of the original image
cv2.imshow('BoundingBox', cv2.flip(annotated_image, 1))
save_image(annotated_image, img_path, annotated_img_path)
