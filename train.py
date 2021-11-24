# ToDo: Update requirements.txt at the end of project
# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os

import keras.layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from glob import glob
import matplotlib.pyplot as plt

import keras
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Lambda, Dense, Flatten, Conv1D, Dropout, MaxPool1D, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Add, Input, ZeroPadding2D, AveragePooling2D,GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import keras.optimizers
import tensorflow as tf
import plots


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

chkp_filepath = 'dataset/saved_model/checkpoints'  # input("Enter the filename you want your model to be saved as: ")
train_path = 'dataset/Image'  # input("Enter the directory of the training images: ")
valid_path = 'dataset/Image'  # input("Enter the directory of the validation images: ")

epochs = 30
batch_size = 32
image_size = 256
IMAGE_SIZE = [image_size, image_size]       # re-size all the images to this
save_trained_model = True


def get_num_of_classes():
	return len(glob(train_path + '/*'))

# load image data and convert it to the right dimensions to train the model. Image data augmentation is uses to generate training data
def load_images():
    train_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)  # rescale=1./255 to scale colors to values between [0,1]
    test_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_generator = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size, subset='training') #, class_mode='categorical')
    valid_generator = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size, subset='validation') #, class_mode='categorical')
    test_generator = test_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size, subset='validation') #, class_mode='categorical') # wird im moment noch nicht benutzt

    return train_generator, valid_generator, test_generator


# Train the model
def train_model(model, train_generator, valid_generator):
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]       # used to save checkpoints during training after each epoch

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs,
                  steps_per_epoch=trainings_samples // batch_size, validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list,
                # steps_per_epoch=len(trainings_samples),validation_steps=len(validation_samples))

    return r, model


# Build the model by transfer learning. This is done by using a pretrained network for feature extraction (DenseNet121)
# and adding a preprocessing layer to adapt to our image dimensions and output layer for our custom number of classes
def create_pretrained_model_densenet121():
    '''
    source: https://github.com/tshr-d-dragon/Sign_Language_Gesture_Detection/blob/main/DenseNet121_MobileNetv2_10epochs.ipynb
    '''
    # ToDo: search in literature for suitable model architectures

    # add preprocessing layer to the front of VGG
    vgg = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    num_of_classes = get_num_of_classes()

    # output layers - you can add more if you want
    x = Flatten()(vgg.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(num_of_classes, activation='softmax', name='predictions')(x)

    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)

    # tell the model what cost and optimization method to use
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_pretrained_model_vgg():
    num_of_classes = get_num_of_classes()
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    VGG = tf.keras.Sequential()
    VGG.add(VGG16(weights='imagenet', include_top=False, input_shape=[image_size, image_size, 3]))
    VGG.add(Flatten())
    VGG.add(Dense(256, activation='relu'))
    VGG.add(Dense(num_of_classes, activation='softmax'))
    VGG.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 2d cnn model for classifying image data
def create_custom_model_2d_cnn():
    num_of_classes = get_num_of_classes()
    model = Sequential()

    # We are using 4 convolution layers for feature extraction
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=[image_size, image_size, 3], kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    # consider using Dropout layers to prevent overfitting
    model.add(Dropout(0.2))  # This is the dropout layer. It's main function is to inactivate 20% of neurons in order to prevent overfitting
    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # We use MaxPooling with a filter size of 2x2. This contributes to generalization

    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))  # , kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # The prevous step gives an output of multi dimentional data, which cannot be fead directly into the feed forward neural network. Hence, the model is flattened
    model.add(Flatten())
    # One hidden layer of 2048 neurons have been used in order to have better classification results    # ToDo: compare classification results for different sizes of hidden layer
    model.add(Dense(2048))  # , kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # The final neuron HAS to be of the same number as classes to predict and cannot be more than that.
    model.add(Dense(num_of_classes, activation='softmax'))  # , activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 2d cnn model for classifying image data
def create_custom_model_2d_cnn_v2():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=[image_size, image_size, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = tf.optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# Save the models and weight for future purposes
def save_model(model):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # serialize model to JSON
    model_json = model.to_json()
    with open("dataset/saved_model/model" + timestr + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("dataset/saved_model/model.h5")
    print("\nSaved model to disk")


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    timestr = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()

    # load image data
    train_generator, valid_generator, test_generator = load_images()

    # load model that uses transfer learning
    model_ = create_pretrained_model_densenet121()
    # load model that uses custom architecture
    # model_ = create_model_2d_cnn()

    # creates the file model_plot.png with a diagram of the created modelmodel structure
    plot_model(model_, to_file='dataset/diagrams/model_plot_' + timestr + '.png', show_shapes=True, show_layer_names=True)  # , rankdir='LR')  # for horizontal direction
    # View the structure of the model
    # model_.summary()

    # Train the model
    history, model = train_model(model_, train_generator, valid_generator)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    if save_trained_model:
        save_model(model)

    # Plot the model Accuracy graph
    plots.plot_training_history(history)
    plots.plot_metrics(history)
