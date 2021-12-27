# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import preprocessing as prep
import plots


test_directory = 'dataset/asl_alphabet_split/test'                                      # define directory of unseen test data

class_labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I',
                'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R',
                'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown']


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # load test images
    test_generator = prep.load_test_images(test_directory)

    # load and create latest created model
    #model = prep.load_latest_model()
    model = prep.load_model_from_name("dataset/saved_model/20211219-180854-pretrained_model_vgg-dataset_asl_alphabet_split")  #

    # tell the model what cost and optimization method to use
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
    # num_of_test_samples = test_generator.samples
    steps_per_epoch = test_generator.n // test_generator.batch_size

    # Generate predictions for samples
    predictions = model.predict(test_generator, steps=steps_per_epoch, verbose=1)

    # plot Confusion Matrix and Classification Report
    plots.plot_confusion_matrix(test_generator.classes, predictions, class_labels)  # true_classes, predicted_classes, labels_of_classes

    # plot multiclass ROC curve with
    plots.plot_roc(test_generator.classes, predictions, class_labels)
