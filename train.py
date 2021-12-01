# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from glob import glob
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import plots
import models
import matplotlib as plt


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

chkp_filepath = 'dataset/saved_model/checkpoints'   # Enter the filename you want your model to be saved as
train_path = 'dataset/asl_alphabet_train'                        # Enter the directory of the training images

epochs = 1
batch_size = 64
image_size = 75
IMAGE_SIZE = [image_size, image_size]               # re-size all the images to this
save_trained_model = False
color_mode = 'grayscale'  # 'rgb'


def get_num_of_classes():
    return len(glob(train_path + '/*'))


# ToDo: get grayscale as image preprocessing function working
def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    plt.imshow(image)
    plt.show()
    return image


# load image data and convert it to the right dimensions to train the model. Image data augmentation is uses to generate training data
def load_training_images():
    train_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2) #, preprocessing_function=to_grayscale_then_rgb)  # , label_mode='categorical')  # rescale=1./255 to scale colors to values between [0,1]
    train_generator = train_gen.flow_from_directory(train_path,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode=color_mode,
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    subset='training',
                                                    class_mode='categorical')

    valid_generator = train_gen.flow_from_directory(train_path,
                                                    target_size=IMAGE_SIZE,
                                                    color_mode=color_mode,
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    subset='validation',
                                                    class_mode='categorical')

    class_occurences = dict(zip(*np.unique(train_generator.classes, return_counts=True)))
    print("class_occurences: \t" + str(class_occurences))

    return train_generator, valid_generator


# Train the model
def train_model(model, train_generator, valid_generator):
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]       # used to save checkpoints during training after each epoch

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    train_class_weights = dict(zip(np.unique(train_generator.classes), class_weights))

    print("train_class_weights: \t" + str(train_class_weights))

    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, class_weight=train_class_weights,
                  steps_per_epoch=trainings_samples // batch_size, validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list,

    return r, model


# Save the models and weight for future purposes
def save_model(model, detailed_model_name):
    model_directory = "dataset/saved_model/"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_directory + detailed_model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_directory + detailed_model_name + ".h5")
    print("\nSaved model to disk")


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    timestr = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()

    # load image data
    train_generator, valid_generator = load_training_images()

    # load model that uses transfer learning
    model_, model_name = models.create_pretrained_model_inception_v3()

    # load model that uses custom architecture
    # model_, model_name = models.create_custom_model_1d_cnn()
    # model_, model_name = models.create_custom_model_2d_cnn_v2()

    # View the structure of the model
    # model_.summary()

    # Train the model
    history, model = train_model(model_, train_generator, valid_generator)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    detailed_model_name = timestr \
                          + "-" + model_name \
                          + "-num_epochs_" + str(epochs) \
                          + "-batch_size_" + str(batch_size) \
                          + "-image_size_" + str(image_size) \
                          + "-acc_" + str(round(history.history['accuracy'][-1], 4)) \
                          + "-val_acc_" + str(round(history.history['val_accuracy'][-1], 4))

    if save_trained_model:
        save_model(model, detailed_model_name)
        # plots.plot_model_structure(model, detailed_model_name)

    # View the structure of the model
    #  model_.summary()

    plot_directory = "dataset/plots/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Plot the model accuracy graph
    plots.plot_training_history(history, plot_directory + detailed_model_name)
    # Plot the model accuracy and loss metrics
    plots.plot_metrics(history, plot_directory + detailed_model_name)

