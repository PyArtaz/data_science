# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import model_from_json
from train import load_training_images
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plots

import matplotlib.pyplot as plt
import preprocessing as prep


test_directory = '../workspace/'
# Model file path
filepath = 'dataset/saved_model/'
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I',
          'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R',
          'S', 'space', 'T', 'U', 'unknown', 'V', 'W', 'X', 'Y', 'Z']  # missing: 'AE', 'OE', 'UE', 'SCH'

# load test images
test_generator = prep.load_test_images(test_directory)

# if you forget to reset the test_generator you will get outputs in a weird order
test_generator.reset()

# load and create latest created model
model = prep.load_latest_model()

# tell the model what cost and optimization method to use
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# division by the number of images in each subfolder provides one classification for all images
steps_per_epoch = test_generator.n // test_generator.batch_size

# Generate predictions for samples
predictions = model.predict(test_generator, steps=steps_per_epoch, verbose=1)

num_of_test_samples = test_generator.samples
batch_size = 32

# Confution Matrix and Classification Report
predicted_categories = tf.argmax(predictions, axis=1)
cm = confusion_matrix(test_generator.classes, predicted_categories)

print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print('Classification Report')
print(classification_report(test_generator.classes, predicted_categories, target_names=labels))

