import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import biosppy
import cv2
from keras.models import model_from_json
import glob
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch
import tensorflow as tf

image_size = 256  # 256
IMAGE_SIZE = [image_size, image_size]               # re-size all the images to this
sampling_rate = 300


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


def preprocess_ecg_leads(ecg_leads_):
    ecg_leads_list = list()

    # create lists containing all data from all csv-files
    for lead in ecg_leads_:
        ecg_leads_list.append(np.array(lead).astype(np.float32))

    # convert lists to numpy arrays
    ecg_leads_array = np.array(ecg_leads_list)  # .astype(np.float32)
    print("ecg_leads_array.shape \t" + str(ecg_leads_array.shape))

    # fills variable length tensors with zeros until all tensors have equal dimensions as he longest sequence
    X_test_tensor = pad_sequence([torch.tensor(x) for x in ecg_leads_array], batch_first=True)

    X_test_tensor = np.expand_dims(X_test_tensor, -1)

    print(X_test_tensor.shape)

    return X_test_tensor


def train_test_split_ecg_leads(ecg_leads_, ecg_labels_):
    ecg_leads_list = list()
    labels_list = list()

    #categorical_to_numerical = {"N": 0, "A": 1, "O": 2, "~": 3}

    # create lists containing all data from all csv-files
    for lead in ecg_leads_:
        ecg_leads_list.append(np.array(lead).astype(np.float32))
    for label in ecg_labels_:
        labels_list.append(np.array(label))

    # convert lists to numpy arrays
    ecg_leads_array = np.array(ecg_leads_list)  # .astype(np.float32)
    ecg_labels_array = np.array(labels_list)  # .astype(np.float32)
    print("ecg_leads_array.shape \t" + str(ecg_leads_array.shape))
    print("ecg_labels_array.shape \t" + str(ecg_labels_array.shape))

    # convert categorical labels into numerical
    _, ecg_labels_array = np.unique(ecg_labels_array, return_inverse=True)  # Note: first variable _ is unused

    X_train, X_test, y_train, y_test = train_test_split(ecg_leads_array, ecg_labels_array, test_size=0.2, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # fills variable length tensors with zeros until all tensors have equal dimensions as he longest sequence
    X_train_tensor = pad_sequence([torch.tensor(x) for x in X_train], batch_first=True)
    X_test_tensor = pad_sequence([torch.tensor(x) for x in X_test], batch_first=True)
    # converts numpy-arrays into tensors
    y_train_tensor = tf.convert_to_tensor(np.asarray(y_train).astype(np.float32), dtype=tf.int32)
    y_test_tensor = tf.convert_to_tensor(np.asarray(y_test).astype(np.float32), dtype=tf.int32)

    X_train_tensor = np.expand_dims(X_train_tensor, -1)
    X_test_tensor = np.expand_dims(X_test_tensor, -1)
    y_train_tensor = np.expand_dims(y_train_tensor, -1)
    y_test_tensor = np.expand_dims(y_test_tensor, -1)

    print(X_train_tensor.shape, X_test_tensor.shape)
    print(y_train_tensor.shape, y_test_tensor.shape)

    return X_train_tensor, X_test_tensor , y_train_tensor, y_test_tensor


def segmentation(path_):
    csv_data = loadmat(path_)
    data = np.array(csv_data['val'][0])
    signals = []
    count = 2
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=sampling_rate)[0]
    for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
        diff1 = abs(peaks[count - 2] - i)
        diff2 = abs(peaks[count + 2] - h)
        x = peaks[count - 2] + diff1 // 2
        y = peaks[count + 2] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 3
    return signals


def segmentation_ecg_lead(ecg_leads, fs):
    data = ecg_leads
    signals = []
    count = 2
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=fs)[0]
    for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
        diff1 = abs(peaks[count - 2] - i)
        diff2 = abs(peaks[count + 2] - h)
        x = peaks[count - 2] + diff1 // 2
        y = peaks[count + 2] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 3
    return signals


def segment_to_single_labeled_img(array_, directory_, filename_, label_):
    new_file_directory = directory_ + '/' + label_ + '/' + filename_ + '/'
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)

        fig = plt.figure(frameon=False)
        plt.plot(array_)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        count = 1

        new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
        fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

        # downsampling images to desired image_size
        im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (image_size, image_size)) #, interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4)
        cv2.imwrite(new_filepath, im_gray)


def segment_to_multiple_labeled_images(array_, directory_, filename_, label_):
    new_file_directory = directory_ + '/' + label_ + '/' + filename_ + '/'
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)

    for count, i in enumerate(array_):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
        fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

        # downsampling images to desired image_size
        im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (image_size, image_size)) #, interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4)
        cv2.imwrite(new_filepath, im_gray)

    return new_file_directory


def segment_to_single_test_img(ecg_segments, ecg_name_, directory_):
    new_file_directory = directory_ + '/' + ecg_name_ + '/'
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)

        if len(ecg_segments) > 0:
            middle_segment = int(len(ecg_segments) // 2)

            # processes just a single three r-peak ecg segment from the middle of the signal
            ecg_segments = ecg_segments[middle_segment]

            fig = plt.figure(frameon=False)
            plt.plot(ecg_segments)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            count = 1

            new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
            fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

            # downsampling images to desired image_size
            im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (image_size, image_size)) #, interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4)
            cv2.imwrite(new_filepath, im_gray)

            print("preprocessing successful: " + ecg_name_)

        else:
            print("preprocessing failed: " + ecg_name_)

            # save the filename of the errorneous ecg_lead in a logging file
            # open the file in the write mode
            # with open('dataset/errorneous_ecg_leads.csv', 'a') as f:
            #     # create the csv writer
            #     writer = csv.writer(f)
            #
            #     # write a row to the csv file
            #     writer.writerow(new_file_directory)
            #
            #     # close the file
            #     f.close()

        return directory_


def segment_to_multiple_test_images(ecg_segments, ecg_name_, directory_):
    new_file_directory = directory_ + '/' + ecg_name_ + '/'
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)

    for count, i in enumerate(ecg_segments):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
        fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

        # downsampling images to desired image_size
        im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (image_size, image_size))  # , interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: choose correct interpolation method
        cv2.imwrite(new_filepath, im_gray)

    return directory_