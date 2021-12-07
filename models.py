# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob
import keras
import keras.layers
from keras.models import Model, Sequential
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten, Conv1D, Dropout, MaxPool1D, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Add, Input, ZeroPadding2D, AveragePooling2D,GlobalAveragePooling2D
import keras.optimizers
import tensorflow as tf

import preprocessing as prep
train_path = prep.train_path
image_size = prep.image_size
IMAGE_SIZE = prep.IMAGE_SIZE               # re-size all the images to this

crossentropy = 'categorical_crossentropy'
activation = 'softmax'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # tell the model what cost and optimization method to use
# optimizer = tf.optimizers.SGD(learning_rate=0.001, momentum=0.9)  # , decay=0.01          # ToDo: try different optimizers

def get_num_of_classes():
    return len(glob(train_path + '/*'))


# Build the model by transfer learning. This is done by using a pretrained network for feature extraction (DenseNet121)
# and adding a preprocessing layer to adapt to our image dimensions and output layer for our custom number of classes
def create_pretrained_model_densenet121():
    '''
    source: https://github.com/tshr-d-dragon/Sign_Language_Gesture_Detection/blob/main/DenseNet121_MobileNetv2_10epochs.ipynb
    '''
    # ToDo: search in literature for suitable model architectures

    # add preprocessing layer to the front of VGG
    densenet = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in densenet.layers:
        layer.trainable = False

    #for i, layer in enumerate(densenet.layers):
    #    print(i, layer.name, layer.trainable)

    num_of_classes = get_num_of_classes()

    # output layers - you can add more if you want
    x = Flatten()(densenet.output)
    x = Dense(512, activation='relu')(x)        # 1000
    prediction = Dense(num_of_classes, activation=activation, name='predictions')(x)

    # create a model object
    model = Model(inputs=densenet.input, outputs=prediction)

    # tell the model what cost and optimization method to use
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model, 'pretrained_model_densenet121'

# Loading pretrained vgg network for transfer learning
'''
Source: https://github.com/krishnasahu29/SignLanguageRecognition/blob/main/vgg16.ipynb
'''
def create_pretrained_model_vgg():
    num_of_classes = get_num_of_classes()

    model = VGG16(weights='imagenet', include_top=False, input_shape=[image_size, image_size, 3])

    # mark loaded layers as not trainable
    for layer in model.layers[:-1]:
        layer.trainable = False

    #for i, layer in enumerate(model.layers):
    #    print(i, layer.name, layer.trainable)

    # add new classifier layers
    flat = Flatten()(model.layers[-1].output)
    #dense1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat)  # 128
    #drop1 = Dropout(0.5)(dense1)
    dense = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat)  # (flat) # (drop1)      # 128
    #drop2 = Dropout(0.5)(dense)
    output = Dense(num_of_classes, activation=activation)(dense)  # (dense)  # (drop2)

    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    model.compile(optimizer=optimizer, loss=crossentropy, metrics=['accuracy'])

    return model, 'pretrained_model_vgg'


# Loading pretrained inception v3 network for transfer learning
'''
Source: https://github.com/VedantMistry13/American-Sign-Language-Recognition-using-Deep-Neural-Network/blob/master/American_Sign_Language_Recognition.ipynb
'''
def create_pretrained_model_inception_v3():
    WEIGHTS_FILE = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    num_of_classes = get_num_of_classes()

    inception_v3_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')

    # Not required --> inception_v3_model.load_weights(WEIGHTS_FILE)

    # Enabling the top 2 inception blocks to train
    for layer in inception_v3_model.layers[:249]:
        layer.trainable = False
    for layer in inception_v3_model.layers[249:]:
        layer.trainable = True

    for i, layer in enumerate(inception_v3_model.layers):
        print(i, layer.name, layer.trainable)

    # Choosing the inception output layer:

    # Choosing the output layer to be merged with our FC layers (if required)
    inception_output_layer = inception_v3_model.get_layer('mixed7')
    # print('Inception model output shape:', inception_output_layer.output_shape)

    # Not required --> inception_output = inception_output_layer.output
    inception_output = inception_v3_model.output

    # Inception model output shape: (None, 10, 10, 768)
    # Adding our own set of fully connected layers at the end of Inception v3 network:
    from tensorflow.keras.optimizers import RMSprop, Adam, SGD

    x = layers.GlobalAveragePooling2D()(inception_output)
    x = layers.Dense(256, activation='relu')(x)
    # Not required --> x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_of_classes, activation=activation)(x)

    model = Model(inception_v3_model.input, x)
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model, 'pretrained_model_inception_v3'


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
    #model.add(Flatten())
    ## One hidden layer of 2048 neurons have been used in order to have better classification results    # ToDo: compare classification results for different sizes of hidden layer
    #model.add(Dense(2048))  # , kernel_initializer='normal', activation='relu'))
    #model.add(keras.layers.ELU())
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    ## The final neuron HAS to be of the same number as classes to predict and cannot be more than that.
    #model.add(Dense(num_of_classes, activation='softmax'))  # , activation='sigmoid'))

    # Final layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_of_classes, activation=activation))

    model.compile(loss=crossentropy, optimizer='adam', metrics=['accuracy'])
    return model, 'custom_model_2d_cnn'


# 2d cnn model for classifying image data
def create_custom_model_2d_cnn_v2():
    num_of_classes = get_num_of_classes()
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

'''
Source: https://stackoverflow.com/questions/60295760/detecting-and-tracking-the-human-hand-with-opencv
'''
def create_custom_model_2d_cnn_v3():
    num_of_classes = get_num_of_classes()

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
