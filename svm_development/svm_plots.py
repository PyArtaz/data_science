import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


# Plot normalized & non-normalized confusion matrix
def plot_cm(model, X_test, y_test):
    titles_options = [
        ("Normalized confusion matrix", "true"),
        ("Confusion matrix without normalization", None),
    ]

    # Generate confusion matrices
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize, xticks_rotation=60)
        disp.ax_.set_title(title)

        # maximize plots to fullscreen. May not work on linux
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.rcParams.update({'font.size': 7})
        plt.tight_layout()
        plt.subplots_adjust(top=0.972, bottom=0.073, left=0.008, right=0.992, hspace=0.2, wspace=0.2)
    plt.show()


# plot the receiver operating characteristic curve
def plot_multiclass_roc(clf, X_test, y_test, class_labels, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic curve')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], class_labels[i]))
    plt.legend(loc = 'lower right', prop = {'size': 7})
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
