import os
from keras.models import model_from_json
import glob
from keras.preprocessing.image import ImageDataGenerator

import train
train_path = train.train_path
image_size = train.image_size
IMAGE_SIZE = train.IMAGE_SIZE


def load_latest_model():
    # File path containing saved models
    filepath = 'dataset/saved_model/'
    # necessary to load the latest saved model in the model folder
    list_of_files = glob.glob(filepath + '*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getmtime)
    head, tail = os.path.split(latest_file)
    model_name = tail.split('.h5')[0]

    # load json and create model
    json_file = open(filepath + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filepath + model_name + '.h5')
    print("Loaded model from disk")

    return loaded_model


def load_model_from_name(model_name):
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    print("Loaded model from disk")

    return loaded_model


# load image data and convert it to the right dimensions to test the model on unseen data
def load_test_images(valid_path):
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, color_mode='rgb', shuffle=False, class_mode=None,
                                                  batch_size=1)  # , class_mode='categorical') # wird im moment noch nicht benutzt  # ToDo: use color_mode='grayscale'
    return test_generator
