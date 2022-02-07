# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cnn_plots
import cnn_models
import pandas as pd
import util

# activate for GPU acceleration
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


################################################################################################################################################################
# Training parameters
################################################################################################################################################################
dataset_path = util.dataset_path                    # Enter the directory of the training images

epochs = 50
batch_size = 32
image_size = util.image_size
IMAGE_SIZE = util.IMAGE_SIZE    # re-size all the images to this
save_trained_model = True

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
    df = df.sample(frac=1).reset_index(drop=True)   # To shuffle the data
    df = df.head(num_train_images)                  # take the smaller subset of data
    print(df)

    return df


# load image data and convert it to the right dimensions to train the model. Image data augmentation is uses to generate more variability in training data
def load_training_images():
    train_gen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   brightness_range=[0.8, 1.2],
                                   fill_mode='constant')  # ,  horizontal_flip=True , label_mode='categorical')

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
                                                        color_mode='rgb',
                                                        shuffle=True)

        valid_generator = val_gen.flow_from_dataframe(directory=dataset_path + '/val',
                                                      dataframe=df,
                                                      x_col="Images",
                                                      y_col="Labels",
                                                      class_mode="categorical",
                                                      batch_size=batch_size,
                                                      target_size=IMAGE_SIZE,
                                                      color_mode='rgb',
                                                      shuffle=True)
    else:
        train_generator = train_gen.flow_from_directory(dataset_path + '/train',
                                                        target_size=IMAGE_SIZE,
                                                        color_mode='rgb',
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
                                                        # save_to_dir=(dataset_path + '/train'+'_augmented'))

        valid_generator = val_gen.flow_from_directory(dataset_path + '/val',
                                                      target_size=IMAGE_SIZE,
                                                      color_mode='rgb',
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

    return train_generator, valid_generator


# initialize checkpoints to save model weights after each epoch if the model's validation accuracy improved.
# EarlyStopping stops training if the model does not longer improve over several epochs
def create_checkpoints():
    # Enter the directory for saving checkpoints during training
    chkp_directory = 'saved_models/checkpoints'
    util.create_folder(chkp_directory)

    # used to save checkpoints during training after each epoch
    checkpoint = ModelCheckpoint(chkp_directory, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # simple early stopping
    if use_reduced_dataset:
        es = EarlyStopping(monitor='accuracy', patience=8, min_delta=0.1, mode='max', restore_best_weights=True)
    else:
        # val_acc has to improve by at least 0.1 for it to count as an improvement
        es = EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.1, mode='max', restore_best_weights=True)
        # es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

    callbacks_list = [checkpoint, es]

    return callbacks_list


# Train the model
def train_model(model, train_generator, valid_generator):
    # deactivated below to prevent unnecessary savings during model optimization phase
    # callbacks_list = create_checkpoints()

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    # calculate class_occurences of dataset
    class_occurences = dict(zip(*np.unique(train_generator.classes, return_counts=True)))
    print("\nclass occurences: \n" + str(class_occurences))

    # calculate class_weights of unbalanced data
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_generator.classes),
                                                      y=train_generator.classes)
    train_class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
    print("\nResulting class weights for training: \n" + str(train_class_weights))

    history = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, class_weight=train_class_weights,
                  steps_per_epoch=trainings_samples // batch_size,
                  validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list)

    return history, model


# create a dictionary containing the timestamp, model name and additional training parameters
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


# update the logging dictionary containing the model's training parameters with its training metrics
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
    model_, model_name = cnn_models.create_pretrained_model_vgg()

    # alternatively load model that uses custom architecture
    # model_, model_name = cnn_models.create_custom_model_2d_cnn_v2()

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
    detailed_model_name = util.create_model_name(log_dict)

    # save trained model and logfile
    if save_trained_model:
        util.save_model_log(log_dict, detailed_model_name)
        util.save_model(model, detailed_model_name)

    # only plot statistics if whole dataset is used, otherwise the metrics val_acc & val_loss are missing in the model's history
    if not use_reduced_dataset:
        # Plot the model accuracy graph
        cnn_plots.plot_training_history(history)
