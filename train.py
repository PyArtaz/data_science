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
from keras.callbacks import ModelCheckpoint, EarlyStopping
import plots
import models
import matplotlib as plt
import preprocessing as prep
import pandas as pd


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

chkp_filepath = 'dataset/saved_model/checkpoints'   # Enter the filename you want your model to be saved as
train_path = prep.train_path                        # Enter the directory of the training images

epochs = 50
batch_size = 32
image_size = prep.image_size
IMAGE_SIZE = prep.IMAGE_SIZE               # re-size all the images to this
save_trained_model = True
color_mode = 'rgb'  # 'grayscale'  # 'rgb'   # mit color_mode='grayscale' passen die 1 dimensionalen Bilder nicht mehr zur vortrainierten Modellarchitektur für rgb Bilder

use_reduced_dataset = False
num_train_images = 100

# ToDo: get grayscale as image preprocessing function working
def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    plt.imshow(image)
    plt.show()
    return image

# reduces the original training dataset and only loads as much "num_train_images" as given. This function is only for faster training purposes
'''
Source: https://stackoverflow.com/questions/58116359/is-there-a-simple-way-to-use-only-half-of-the-images-in-underlying-directories-u
'''
def subset_training_images(num_train_images):
    import os
    images = []
    labels = []
    for sub_dir in os.listdir(train_path):
        image_list = os.listdir(os.path.join(train_path,sub_dir))  # list of all image names in the directory
        image_list = list(map(lambda x:os.path.join(sub_dir,x),image_list))
        images.extend(image_list)
        labels.extend([sub_dir]*len(image_list))

    df = pd.DataFrame({"Images": images, "Labels": labels})
    df = df.sample(frac=1).reset_index(drop=True)  # To shuffle the data # ToDo: subset df with equal class occurences
    df = df.head(num_train_images)  # to take the subset of data (I'm taking 100 from it)
    print(df)

    return df

# load image data and convert it to the right dimensions to train the model. Image data augmentation is uses to generate training data
def load_training_images():
    # ToDo: find good parameters. fill_mode nearest produces strange results with fingers
    train_gen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   brightness_range=[0.8, 1.2],
                                   fill_mode='nearest',
                                   validation_split=0.2)  # brightness_range=[0.8, 1.2], # horizontal_flip=True,# , preprocessing_function=to_grayscale_then_rgb)  # , label_mode='categorical')  # rescale=1./255 to scale colors to values between [0,1]

    val_gen = ImageDataGenerator(rescale=1. / 255.,
                                 validation_split=0.2)

    if use_reduced_dataset:
        df = subset_training_images(num_train_images=num_train_images)
        train_generator = train_gen.flow_from_dataframe(directory=train_path,
                                                        dataframe=df,
                                                        x_col="Images",
                                                        y_col="Labels",
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        target_size=IMAGE_SIZE,
                                                        color_mode=color_mode,
                                                        shuffle=True,
                                                        subset='training')

        # ToDo: find error why val_acc & val_loss are not computed when using flow_from_dataframe
        valid_generator = val_gen.flow_from_dataframe(directory=train_path,
                                                      dataframe=df,
                                                      x_col="Images",
                                                      y_col="Labels",
                                                      class_mode="categorical",
                                                      batch_size=batch_size,
                                                      target_size=IMAGE_SIZE,
                                                      color_mode=color_mode,
                                                      shuffle=True,
                                                      subset='validation')
    else:
        train_generator = train_gen.flow_from_directory(train_path,
                                                        target_size=IMAGE_SIZE,
                                                        color_mode=color_mode,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        subset='training',
                                                        class_mode='categorical',
                                                        save_to_dir=(train_path+'_augmented'))

        valid_generator = val_gen.flow_from_directory(train_path,
                                                      target_size=IMAGE_SIZE,
                                                      color_mode=color_mode,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      subset='validation',
                                                      class_mode='categorical')

    return train_generator, valid_generator


# Train the model
def train_model(model, train_generator, valid_generator):
    # used to save checkpoints during training after each epoch
    checkpoint = ModelCheckpoint(chkp_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # simple early stopping
    es_acc = EarlyStopping(monitor='accuracy', patience=8, min_delta=0.1, mode='max', restore_best_weights=True)
    es_val_loss = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
    es_val_acc  = EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.01, mode='max', restore_best_weights=True)  # val_acc has to improve by at least 0.1 for it to count as an improvement
    callbacks_list = [checkpoint, es_acc]

    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    # calculate class_occurences of dataset
    class_occurences = dict(zip(*np.unique(train_generator.classes, return_counts=True)))
    print("class_occurences: \t" + str(class_occurences))

    # calculate class_weights of unbalanced data
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    train_class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
    print("train_class_weights: \t" + str(train_class_weights))

    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, class_weight=train_class_weights,
                  steps_per_epoch=trainings_samples // batch_size, validation_steps=validation_samples // batch_size)  # , callbacks=callbacks_list)  # , callbacks=callbacks_list,

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
    model_, model_name = models.create_pretrained_model_vgg()

    # load model that uses custom architecture
    # model_, model_name = models.create_custom_model_2d_cnn_v2()
    # model_, model_name = models.create_custom_model_2d_cnn_v3()

    # View the structure of the model
    # model_.summary()

    print("Begin training of: "
          + timestr
          + "-" + model_name
          + "-dataset_" + prep.dataset
          + "-num_epochs_" + str(epochs)
          + "-batch_size_" + str(batch_size)
          + "-image_size_" + str(image_size))

    # Train the model
    history, model = train_model(model_, train_generator, valid_generator)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    if use_reduced_dataset:
        detailed_model_name = timestr \
                              + "-" + model_name \
                              + "-dataset_" + prep.dataset \
                              + "-num_epochs_" + str(epochs) \
                              + "-batch_size_" + str(batch_size) \
                              + "-image_size_" + str(image_size) \
                              + "-acc_" + str(round(history.history['accuracy'][-1], 4)).replace('.', '_') \
                              + "-loss_" + str(round(history.history['loss'][-1], 4)).replace('.', '_')
    else:
        detailed_model_name = timestr \
                              + "-" + model_name \
                              + "-dataset_" + prep.dataset \
                              + "-num_epochs_" + str(epochs) \
                              + "-batch_size_" + str(batch_size) \
                              + "-image_size_" + str(image_size) \
                              + "-acc_" + str(round(history.history['accuracy'][-1], 4)).replace('.', '_') \
                              + "-loss_" + str(round(history.history['loss'][-1], 4)).replace('.', '_') \
                              + "-val_acc_" + str(round(history.history['val_accuracy'][-1], 4)).replace('.', '_') \
                              + "-val_loss_" + str(round(history.history['val_loss'][-1], 4)).replace('.', '_')

    if save_trained_model:
        save_model(model, detailed_model_name)
        # plots.plot_model_structure(model, detailed_model_name)

    # View the structure of the model
    #  model_.summary()

    plot_directory = "dataset/plots/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # only plot statistics if whole dataset is used, otherwise history is missing val_acc & val_loss
    if not use_reduced_dataset:
        # Plot the model accuracy graph
        plots.plot_training_history(history, plot_directory + detailed_model_name)
        # Plot the model accuracy and loss metrics
        # plots.plot_metrics(history, plot_directory + detailed_model_name)

