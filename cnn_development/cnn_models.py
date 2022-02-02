# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model, Sequential
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, MaxPool2D, Add, ZeroPadding2D
import keras.optimizers
import tensorflow as tf
import util


################################################################################################################################################################
# General Parameters and functions
################################################################################################################################################################

image_size = util.image_size
IMAGE_SIZE = util.IMAGE_SIZE               # re-size all the images to this

crossentropy = 'categorical_crossentropy'
activation = 'softmax'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # tell the model what cost and optimization method to use


################################################################################################################################################################
# Pretrained Models using Transfer Learning
################################################################################################################################################################

# Build the model by transfer learning. This is done by using a pretrained network for feature extraction (DenseNet121)
# and adding a preprocessing layer to adapt to our image dimensions and output layer for our custom number of classes
def create_pretrained_model_densenet121():
    """
    source: https://github.com/tshr-d-dragon/Sign_Language_Gesture_Detection/blob/main/DenseNet121_MobileNetv2_10epochs.ipynb
    """
    # add preprocessing layer to the front of DenseNet121
    densenet = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in densenet.layers:
        layer.trainable = False

    # for i, layer in enumerate(densenet.layers):
    #     print(i, layer.name, layer.trainable)

    num_of_classes = util.get_num_of_classes()

    # output layers
    x = Flatten()(densenet.output)
    x = Dense(512, activation='relu')(x)        # 1000
    prediction = Dense(num_of_classes, activation=activation, name='predictions')(x)

    # create a model object
    model = Model(inputs=densenet.input, outputs=prediction)

    # tell the model what cost and optimization method to use
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model, 'pretrained_model_densenet121'


# Loading pretrained vgg network for transfer learning
def create_pretrained_model_vgg():
    """
    Source: https://github.com/krishnasahu29/SignLanguageRecognition/blob/main/vgg16.ipynb
    """
    num_of_classes = util.get_num_of_classes()

    model = VGG16(weights='imagenet', include_top=False, input_shape=[image_size, image_size, 3])

    # mark loaded layers as not trainable
    for layer in model.layers[:-1]:
        layer.trainable = False

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name, layer.trainable)

    # add new classifier layers
    flat = Flatten()(model.layers[-1].output)
    # dense1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat)
    # drop1 = Dropout(0.5)(dense1)
    dense = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat)
    # drop2 = Dropout(0.5)(dense)
    output = Dense(num_of_classes, activation=activation)(dense)

    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    model.compile(optimizer=optimizer, loss=crossentropy, metrics=['accuracy'])

    return model, 'pretrained_model_vgg'


# Loading pretrained inception v3 network for transfer learning
def create_pretrained_model_inception_v3():
    """
    Source: https://github.com/VedantMistry13/American-Sign-Language-Recognition-using-Deep-Neural-Network/blob/master/American_Sign_Language_Recognition.ipynb
    """
    num_of_classes = util.get_num_of_classes()

    inception_v3_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')

    # Enabling the top two inception blocks to train
    for layer in inception_v3_model.layers[:249]:
        layer.trainable = False
    for layer in inception_v3_model.layers[249:]:
        layer.trainable = True

    for i, layer in enumerate(inception_v3_model.layers):
        print(i, layer.name, layer.trainable)

    # Choosing the inception output layer:

    # Choosing the output layer to be merged with our Fully Connected layers (if required)
    inception_output = inception_v3_model.output

    # Adding our own set of fully connected layers at the end of Inception v3 network:
    x = layers.GlobalAveragePooling2D()(inception_output)
    x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_of_classes, activation=activation)(x)

    model = Model(inception_v3_model.input, x)
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model, 'pretrained_model_inception_v3'


################################################################################################################################################################
# Custom models using self defined structure
################################################################################################################################################################

# 2d cnn model for classifying image data
def create_custom_model_2d_cnn():
    num_of_classes = util.get_num_of_classes()
    model = Sequential()

    # We are using multiple convolution layers for feature extraction
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=[image_size, image_size, 3], kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    # use Dropout layers to prevent overfitting
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

    # Final layer
    model.add(Flatten())
    # Two hidden layers of 4096 and 2048 neurons have been used in order to achieve better classification results
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.1))
    # The final neuron HAS to be of the same number as classes to predict and cannot be more than that.
    model.add(Dense(num_of_classes, activation=activation))

    model.compile(loss=crossentropy, optimizer='adam', metrics=['accuracy'])
    return model, 'custom_model_2d_cnn'


# 2d cnn model for classifying image data
def create_custom_model_2d_cnn_v2():
    num_of_classes = util.get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(128, (2,2), input_shape=[image_size, image_size, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(256, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation=activation))
    sgd = tf.optimizers.SGD(learning_rate=1e-2)
    model.compile(loss=crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model, 'custom_model_2d_cnn_v2'

# 2d cnn model for classifying image data
def create_custom_model_2d_cnn_v3():
    """
    Source: https://stackoverflow.com/questions/60295760/detecting-and-tracking-the-human-hand-with-opencv
    """
    num_of_classes = util.get_num_of_classes()

    i = Input(shape=[image_size, image_size, 3])
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)

    # x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_of_classes, activation='softmax')(x)

    model = Model(i, x)

    model.compile(loss=crossentropy, optimizer=optimizer, metrics=['accuracy'])     # optimizer = 'SGD'

    return model, 'custom_model_2d_cnn_v3'