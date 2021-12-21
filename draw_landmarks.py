from util import draw_landmarks, save_image
import pandas as pd

data = '../data_science/dataset/hand_landmarks/asl_alphabet_train/asl_alphabet_train_landmarks.csv'
img_path = '../data_science/dataset/asl_alphabet_train/A/A2.jpg'
annotated_img_path = '../data_science/dataset/hand_landmarks/evaluation/asl_alphabet_train/post'

# Construct DataFrame from csv file specified by data
df = pd.read_csv(data, header=0, index_col=0)
# Annotate the image specified by img_path with landmarks from the dataframe
annotated_image = draw_landmarks(img_path, df)

# Save the annotated image to the annotated_img_path using the name of the original image
save_image(annotated_image, img_path, annotated_img_path)
