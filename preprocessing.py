import os
import glob
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

###############################
# Dataset parameters
###############################
dataset_path = 'dataset/asl_alphabet'          # Enter the directory containing the training images
model_directory = 'dataset/saved_model/'             # directory to save the model

##############################
# Training parameters
##############################
image_size = 100
IMAGE_SIZE = [image_size, image_size]               # re-size all the images to this


################################################################################################################################################################
# General FUNCTIONS
################################################################################################################################################################
# checks if folder already exists and creates it if not
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_num_of_classes():
    return len(glob.glob(dataset_path + '/train/*'))


################################################################################################################################################################
# LOADING FUNCTIONS
################################################################################################################################################################
def load_latest_model():
    # File path containing saved models
    filepath = 'dataset/saved_model/'
    # necessary to load the latest saved model in the model folder
    list_of_files = glob.glob(filepath + '*.h5')                     # '*' means all if need specific format then e.g.: '*.h5'
    latest_file = max(list_of_files, key=os.path.getmtime)
    head, tail = os.path.split(latest_file)
    model_name = tail.split('.h5')[0]

    # load json and weights and create model
    loaded_model = load_model_from_name(filepath + model_name)

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
    test_generator = test_gen.flow_from_directory(valid_path,
                                                  target_size=IMAGE_SIZE,
                                                  color_mode='rgb',
                                                  shuffle=False,
                                                  batch_size=1,
                                                  class_mode=None)  # , class_mode='categorical') # wird im moment noch nicht benutzt

    # if you forget to reset the test_generator you will get outputs in a weird order
    test_generator.reset()

    return test_generator


################################################################################################################################################################
# SAVING FUNCTIONS
################################################################################################################################################################

def create_model_name(info_dict):
    return info_dict['Time'] + '-' + info_dict['Model name'] + '-dataset_' + info_dict['Used dataset']


# Save a log file to accompany the saved model
def save_model_log(log_dict, model_name):
    create_folder(model_directory)

    with open(model_directory + model_name + '_log.txt', 'w') as log_file:
        for key, value in log_dict.items():
            log_file.write('%s: %s\n' % (key, value))

    print('\nSaved log file to disk')


# Save the models and weight for future purposes
def save_model(model, detailed_model_name):
    create_folder(model_directory)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_directory + detailed_model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_directory + detailed_model_name + ".h5")
    print("\nSaved model to disk")
