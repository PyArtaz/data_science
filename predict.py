import tensorflow as tf
from keras.models import model_from_json
from train import load_images
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# File path
filepath = 'dataset/saved_model/'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'AE', 'OE', 'UE', 'SCH', 'One', 'Two', 'Three', 'Four', 'Five']


# load json and create model
json_file = open(filepath + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(filepath + 'model.h5')
print("Loaded model from disk")

# A few random samples
train_generator, valid_generator, test_generator = load_images()

# Generate predictions for samples
predictions = loaded_model.predict(test_generator)  # , num_of_test_samples // batch_size+1)

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

# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
