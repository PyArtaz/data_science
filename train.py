# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import plots
import models
import matplotlib as plt
import pandas as pd
import preprocessing as prep

# activate for GPU acceleration
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


################################################################################################################################################################
# Training parameters
################################################################################################################################################################
chkp_filepath = 'dataset/saved_model/checkpoints'  # Enter the filename you want your model to be saved as
dataset_path = prep.dataset_path  # Enter the directory of the training images

epochs = 50
batch_size = 32
image_size = prep.image_size
IMAGE_SIZE = prep.IMAGE_SIZE    # re-size all the images to this
save_trained_model = True
color_mode = 'rgb'  # Notiz: mit color_mode='grayscale' passen die 1 dimensionalen Bilder ggf. nicht mehr zur vortrainierten Modellarchitektur für rgb Bilder

use_reduced_dataset = False     # set true to use a reduced dataset with a total amount of num_train_images
num_train_images = 1000


################################################################################################################################################################
# Training FUNCTIONS
################################################################################################################################################################

# reduces the original training dataset and only loads as much "num_train_images" as given. This function is only for faster training purposes
def subset_training_images(num_train_images):
    """
    Source: https://stackoverflow.com/questions/58116359/is-there-a-simple-way-to-use-only-half-of-the-images-in-underlying-directories-u
    """
    images = []
    labels = []
    for sub_dir in os.listdir(dataset_path + '/train'):
        image_list = os.listdir(os.path.join(dataset_path + '/train', sub_dir))  # list of all image names in the directory
        image_list = list(map(lambda x: os.path.join(sub_dir, x), image_list))
        images.extend(image_list)
        labels.extend([sub_dir] * len(image_list))

    df = pd.DataFrame({"Images": images, "Labels": labels})
    df = df.sample(frac=1).reset_index(drop=True)  # To shuffle the data # ToDo: subset df with equal class occurences
    df = df.head(num_train_images)  # to take the subset of data (I'm taking 100 from it)
    print(df)

    return df


# load image data and convert it to the right dimensions to train the model. Image data augmentation is uses to generate training data
def load_training_images():
    # ToDo: find good parameters. fill_mode='nearest' produces strange results with fingers
    train_gen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   brightness_range=[0.8, 1.2],
                                   fill_mode='nearest')  # ,  horizontal_flip=True , label_mode='categorical')

    val_gen = ImageDataGenerator(rescale=1. / 255.)

    if use_reduced_dataset:
        df = subset_training_images(num_train_images=num_train_images)
        train_generator = train_gen.flow_from_dataframe(directory=dataset_path + '/train',
                                                        dataframe=df,
                                                        x_col="Images",
                                                        y_col="Labels",
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        target_size=IMAGE_SIZE,
                                                        color_mode=color_mode,
                                                        shuffle=True)

        # ToDo: find error why val_acc & val_loss are not computed when using flow_from_dataframe
        valid_generator = val_gen.flow_from_dataframe(directory=dataset_path + '/val',
                                                      dataframe=df,
                                                      x_col="Images",
                                                      y_col="Labels",
                                                      class_mode="categorical",
                                                      batch_size=batch_size,
                                                      target_size=IMAGE_SIZE,
                                                      color_mode=color_mode,
                                                      shuffle=True)
    else:
        train_generator = train_gen.flow_from_directory(dataset_path + '/train',
                                                        target_size=IMAGE_SIZE,
                                                        color_mode=color_mode,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
        # save_to_dir=(train_path+'_augmented'))

        valid_generator = val_gen.flow_from_directory(dataset_path + '/val',
                                                      target_size=IMAGE_SIZE,
                                                      color_mode=color_mode,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

    return train_generator, valid_generator


def create_checkpoints():
    # used to save checkpoints during training after each epoch
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # simple early stopping
    if use_reduced_dataset:
        es = EarlyStopping(monitor='accuracy', patience=8, min_delta=0.1, mode='max', restore_best_weights=True)
    else:
        # val_acc has to improve by at least 0.1 for it to count as an improvement
        es = EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.01, mode='max', restore_best_weights=True)
        # es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

    callbacks_list = [checkpoint, es]

    return callbacks_list


# Train the model
def train_model(model, train_generator, valid_generator):
    callbacks_list = create_checkpoints()                               # deactivated below to prevent unnecessary savings during model optimization phase

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    # calculate class_occurences of dataset
    class_occurences = dict(zip(*np.unique(train_generator.classes, return_counts=True)))
    print("class_occurences: \t" + str(class_occurences))

    # calculate class_weights of unbalanced data
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_generator.classes),
                                                      y=train_generator.classes)
    train_class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
    print("train_class_weights: \t" + str(train_class_weights))

    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, class_weight=train_class_weights,
                  steps_per_epoch=trainings_samples // batch_size,
                  validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list)

    return r, model


def plot_statistics(model_name, history):
    plot_directory = "dataset/plots/"
    prep.create_folder(plot_directory)

    # Plot the model accuracy graph
    plots.plot_training_history(history, plot_directory + model_name)


def create_logdict():
    dataset_name = dataset_path.split('/')[-1]        # dataset name used for detailed model name

    # initialize logging of training parameters
    log_dict = {'Time': timestr,
                'Model name': model_name,
                'Used dataset': dataset_name,
                'Number of epochs': epochs,
                'Image size': image_size,
                'Batch size': batch_size}

    return log_dict


def update_logdict():
    # get training metrics
    if use_reduced_dataset:
        after_run_info = {'Run time': pred_time,
                          'Accuracy': round(history.history['accuracy'][-1], 4),
                          'Loss': round(history.history['loss'][-1], 4)}
    else:
        after_run_info = {'Run time': pred_time,
                          'Accuracy': round(history.history['accuracy'][-1], 4),
                          'Loss': round(history.history['loss'][-1], 4),
                          'Validation accuracy': round(history.history['val_accuracy'][-1], 4),
                          'Validation loss': round(history.history['val_loss'][-1], 4)}

    # update logging with training metrics
    log_dict.update(after_run_info)

    return log_dict


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    timestr = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()

    # load image data
    train_generator, valid_generator = load_training_images()

    # load model that uses transfer learning
    model_, model_name = models.create_pretrained_model_vgg()

    # alternatively load model that uses custom architecture
    # model_, model_name = models.create_custom_model_2d_cnn_v2()

    # View the structure of the model
    # model_.summary()

    # save training parameters in a logging dictionary
    log_dict = create_logdict()
    print('\nStart training of: ', log_dict)

    # Train the model
    history, model = train_model(model_, train_generator, valid_generator)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    # update logging dict with training statistics
    log_dict = update_logdict()

    # create more detailed model name to distinguish saved models
    detailed_model_name = prep.create_model_name(log_dict)

    # save trained model and logfile
    if save_trained_model:
        prep.save_model_log(log_dict, detailed_model_name)
        prep.save_model(model, detailed_model_name)
        # save a graph plot of the models layer structure
        # plots.plot_model_structure(model, detailed_model_name)

    # only plot statistics if whole dataset is used, otherwise history is missing val_acc & val_loss
    if not use_reduced_dataset:
        plot_statistics(detailed_model_name, history)
