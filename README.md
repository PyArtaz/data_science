# American sign language alphabet recognition

## Project idea

Camera-equipped edge computing devices such as the Nvidia Jetson Nano used in our project can recognize the sign language alphabet using machine learning. MediaPipe hands extracts the joint positions of a presented hand gesture which is then classified using a support vector machine (SVM) for real-time processing. Training is performed on a powerful computer. A self-captured database and landmark normalization enable robust prediction. The classification is presented in written form on a connected monitor, allowing to spell out words.

![SignumFinal_v2](https://user-images.githubusercontent.com/33580996/152803487-88b3daa1-67fd-45e7-8bed-63e5367528d8.PNG)

### Technology stack

The whole project is programmed using Python 3.9. It is designed to run on a Nvidia Jetson Nano (2 GB RAM) equipped with a camera and connected to a monitor. The GUI is programmed using PyQt, the Python port of the popular Qt framework.

[MediaPipe hands](https://google.github.io/mediapipe/solutions/hands.html) is part of the popular open-source machine learning library [MediaPipe](https://google.github.io/mediapipe/) developed by Google. We use MP hands to extract joint coordinates, our features, from the camera frames. These are stored in a [NumPy](https://numpy.org/) ndarray which is then classified by an SVM trained using our own database. It contains images of all letters of the American sign language alphabet, digits and three custom gestures, e.g., for UI interaction. More on the database can be found below.

The development of a highly optimized machine learning pipeline is accelerated by [TPOT](https://epistasislab.github.io/tpot/), built on top of [scikit-learn](https://scikit-learn.org/stable/). The freely available datasets [ASL alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) and [Sign Language for Numbers](https://www.kaggle.com/muhammadkhalid/sign-language-for-numbers) provide test instances to evaluate the trained SVM, which results in an average F1-score of 78.29 %.


## Repository

This repository consists of two branches, jetnano_final and model_development that both fullfill different purposes.

### jetnano_final

This branch conatains all files necessary to run the final project on your hardware, be it a Nvidia Jetson Nano or a computer running Python. All you have to do is change the value of the jet_nano boolean accordingly. This branch includes:

- models folder: Support vector machines trained on different datasets and saved using [Joblib](https://joblib.readthedocs.io/en/latest/). By default, the own_landmark models for digits, numbers and both are used.

- Python scripts
	- start_gui_v6.py: Start the GUI application for recognition of the gestures presented above.
	- signum_gui_v5.py: The layout file for the GUI obtained by translating the Qt UI file using pyuic5.
	- util.py: Utility file containing many auxiliary functions for loading, saving, annotating and calculating.
	- bounding_box.py: Implementation of the BoundingBox class to calculate a bounding box from the MP hand landmarks.
	- predictor.py: Implementation of the classes Predictor and DataLogger for classification of the hand landmarks and data logging purposes, respectively.
	
### model_development

This branch contains files for data preprocessing of image databases using MediaPipe and files to train support vector machines using the hand landmarks. There even is a possibility to use transfer learning for convolutional neural networks. This branch includes:

- cnn_development folder: Train, test and save pre-trained convolutional neural networks for image classification.

- svm_development folder: Train, test and save support vector machines for classification of hand landmarks.
	- tpot_model_optimization.py: Use the auto-ML tool TPOT to find the optimal machine learning model for your data.
	- svm_plots.py: Utility file to create plots such as a confusion matrix.
	
- preprocessing folder: Create a hand landmarks database from your existing image database using MediaPipe hands and normalization functions


## Our Dataset

There are extensive, open-source databases for both, the sign language and the sign language alphabet (relevant for us). Insufficient variance in existing databases such as the ASL alphabet dataset and the sign language for numbers dataset led us to create our own database. It contains images of all letters of the American sign language alphabet, digits and three custom gestures, e.g., for UI interaction. Therefore, the datset contains 39 classes (Gestures for 26 letters, 10 digits and 'ENTER', 'SPACE' and 'DELETE') recorded by four people with approximately 240 images per class. Examples for each class can be found below.

![A-T](https://user-images.githubusercontent.com/33580996/152804646-00540801-1bfb-4edf-81e2-cbd2a55bbb63.png)
![U-0](https://user-images.githubusercontent.com/33580996/152804638-ea07d66d-fc1a-4236-8048-2bf6b9c67ff4.png)

## Getting Started

First, make sure you have Python 3.8 installed on your device.

1. Clone this repository
2. Best practice: Create a new virtual environment, for example using Anaconda
3. Install project requirements using pip with this command: `pip install -r requirements.txt`
4. You can start the GUI by executing `start_gui_v6.py`

### Create your own application using model_development

Make sure you followed the first three steps above and then switched the branch to model_development.

1. Create your own hand-gesture database using `preprocessing/capture.py` or use an existing database
2. Extract the hand landmarks from the database images using `preprocessing/source_data.py`
3. If you require normalized hand landmarks, run `preprocessing/change_coordinates.py` with the dataset created in step 2
4. Train an SVM using scikit-learn and save it by executing `svm_development/svm_training.py`
5. Evaluate the previously traind SVM using `svm_development/svm_evaluation.py`
