# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from itertools import cycle


# creates a diagram of the model's layer structure
# You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/)
def plot_model_structure(model_, model_name):
    diagram_directory = 'diagrams/'
    if not os.path.exists(diagram_directory):
        os.makedirs(diagram_directory)

    # creates the file model_plot.png with a diagram of the created modelmodel structure
    plot_model(model_, to_file=diagram_directory + model_name + '.png', show_shapes=True, show_layer_names=True)  # , rankdir='LR')  # for horizontal direction


# Plot the model Accuracy graph
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Training and Validation loss')

    # Plot the model Accuracy graph
    ax1.plot(history.history['accuracy'], 'r', linewidth=3.0, label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], 'b', linewidth=3.0, label='Validation Accuracy')
    ax1.legend(fontsize=12)
    ax1.set(xlabel='Epochs ', ylabel='Accuracy')
    ax1.set_title('Accuracy Curves')

    ax1.set_axisbelow(True)                                                     # Don't allow the axis to be on top of your data
    ax1.minorticks_on()                                                         # Turn on the minor TICKS, which are required for the minor GRID
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black')      # Customize the major grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')      # Customize the minor grid

    # Plot the model Loss graph (Ideally it should be Exponentially decreasing shape)
    ax2.plot(history.history['loss'], 'g', linewidth=3.0, label='Training Loss')
    ax2.plot(history.history['val_loss'], 'y', linewidth=3.0, label='Validation Loss')
    ax2.legend(fontsize=12)
    ax2.set(xlabel='Epochs ', ylabel='Loss')
    ax2.set_title('Loss Curves')

    ax2.set_axisbelow(True)
    ax2.minorticks_on()
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    fig.tight_layout()
    plt.show()


# plot the model's Confusion Matrix
def plot_confusion_matrix(classes, predictions, labels):
    predicted_categories = tf.argmax(predictions, axis=1)
    cm = confusion_matrix(classes, predicted_categories)

    # print('Confusion Matrix')
    # print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


# plot the model's receiver operating characteristic curve
def plot_roc(y_true, y_pred, class_labels):
    """
    Source: https://datascience.stackexchange.com/questions/82378/auc-on-roc-curve-near-1-0-for-multi-class-cnn-but-precision-recall-are-not-perfe
    """
    # transforms strings in class_labels in integers for further processing
    integer_class_labels = list(range(len(class_labels)))

    # Binarize the output
    y_true = label_binarize(y_true, classes=integer_class_labels)
    n_classes = y_true.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute fpr and tpr with roc_curve from the ytest true labels to the scores
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot each class  curve on single graph for multi-class one vs all classification
    colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
    for i, color, label in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(label, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    plt.show()
