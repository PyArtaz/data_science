# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ToDo: save plots with filename in format: datetime-model_name-image_resolution-number_of_epochs-batch_size

def plot_model_structure(model_, detailed_model_name):
    if not os.path.exists('dataset/diagrams/'):
        os.makedirs('dataset/diagrams/')
    # creates the file model_plot.png with a diagram of the created modelmodel structure
    plot_model(model_, to_file='dataset/diagrams/' + detailed_model_name + '.png', show_shapes=True, show_layer_names=True)  # , rankdir='LR')  # for horizontal direction


# Plot the model Accuracy graph
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Training and Validation loss')

    # Plot the model Accuracy graph (Ideally, it should be Logarithmic shape)
    ax1.plot(history.history['accuracy'], 'r', linewidth=3.0, label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], 'b', linewidth=3.0, label='Validation Accuracy')
    ax1.legend(fontsize=12)
    ax1.set(xlabel='Epochs ', ylabel='Accuracy')
    ax1.set_title('Accuracy Curves')

    ax1.set_axisbelow(True)                                                     # Don't allow the axis to be on top of your data
    ax1.minorticks_on()                                                         # Turn on the minor TICKS, which are required for the minor GRID
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black')        # Customize the major grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')      # Customize the minor grid

    # Plot the model Loss graph (Ideally it should be Exponentially decreasing shape)
    ax2.plot(history.history['loss'], 'g', linewidth=3.0, label='Training Loss')
    ax2.plot(history.history['val_loss'], 'y', linewidth=3.0, label='Validation Loss')
    ax2.legend(fontsize=12)
    ax2.set(xlabel='Epochs ', ylabel='Loss')
    ax2.set_title('Loss Curves')

    ax2.set_axisbelow(True)                                                     # Don't allow the axis to be on top of your data
    ax2.minorticks_on()                                                         # Turn on the minor TICKS, which are required for the minor GRID
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')        # Customize the major grid
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')      # Customize the minor grid

    fig.tight_layout()
    plt.show()


def plot_metrics(history):
    metrics = ['accuracy', 'loss']  #, 'precision', 'recall']

    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        fig = plt.subplot(1,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.legend()
    plt.show()
