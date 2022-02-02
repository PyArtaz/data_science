import time
from tpot import TPOTClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from svm_training import load_dataframe, print_statistics, save_trained_model
import svm_plots


################################################################################################################################################################
# Training parameters
################################################################################################################################################################
dataset_path = '../dataset/hand_landmarks/Own/Own_landmarks_bb_squarePix_Letters+Digits.csv'  # Enter the directory of the training images

################################################################################################################################################################
# Training FUNCTIONS
################################################################################################################################################################

# Perform an automated search with evolutionary algorithms to find an optimized pipeline
# The performed pipeline optimization includes steps for data preprocessing, model selection and hyperparameter optimization
def perform_tpot_search(X_train, y_train):
    # define search
    tpot = TPOTClassifier(generations=1, population_size=2, cv=10, random_state=42, verbosity=2, n_jobs=-1)

    # perform the search
    tpot.fit(X_train, y_train)

    # export the best model into a separate python script
    timestr = time.strftime("%Y%m%d-%H%M%S")
    tpot.export('tpot_model_' + timestr + '.py')

    return tpot


# perform a grid search to find the optimal hyperparameters for the SVM model
def perform_svc_grid_search(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'degree': [3, 5, 7], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
                  'kernel': ['rbf', 'poly', 'sigmoid']}

    model = SVC(class_weight='balanced', probability=True)

    grid = GridSearchCV(model, param_grid, refit=True, verbose=4, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_)

    return grid


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # define whether you want to search for an optimized model or use a previously found one
    perform_automatic_model_search = True

    start_time = time.time()

    # load hand_landmark data
    X_train, X_test, y_train, y_test = load_dataframe(dataset_path, resampling='over', perform_train_test_split=True)

    # either: search for suitable model automated evolutionary search or by manual grid search
    if perform_automatic_model_search:
        # automated search for an optimized model by TPOT
        model = perform_tpot_search(X_train.values, y_train)
    else:
        # manual grid search for hyperparameter optimization on a SVM model
        model = perform_svc_grid_search(X_train.values, y_train)

    # Train the model
    model.fit(X_train.values, y_train)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    # check performance on unseen test data
    predictions = model.predict(X_test)

    # calculate metrics to evaluate model performance on unseen test data
    print_statistics(y_test, predictions)

    if not perform_automatic_model_search:
        save_trained_model(model)

        # plot confusion matrix after training in unnormalized and normalized fashion
        svm_plots.plot_cm(model, X_test, y_test)
