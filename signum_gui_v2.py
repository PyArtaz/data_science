import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QPushButton
)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
from signum_gui_layout import Ui_MainWindow

import numpy as np
from keras.preprocessing.image import img_to_array
import preprocessing as prep
import time


image_size = prep.image_size
IMAGE_SIZE = prep.IMAGE_SIZE               # re-size all the images to this
bottom_left, bottom_right = 150, 150

model_directory = 'dataset/saved_model/'
model_name = '20211204-200825-pretrained_model_vgg-num_epochs_3-batch_size_32-image_size_64-acc_0_9065-val_acc_0_8569'
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#alphabet = ['A', 'B', 'C', 'del', 'I', 'J', 'S', 'space', 'T', 'Z']

# load and create latest created model
model = prep.load_latest_model()
# model = prep.load_model_from_name(model_directory + model_name)

# tell the model what cost and optimization method to use
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

debug = True


class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        print("Found camera indizes: " + str(self.returnCameraIndexes()))
        
        #Start und Stop Button
        self.ui.StopPlay.clicked.connect(self.VideoStopSlot)
        self.ui.StartPlay.clicked.connect(self.VideoStartSlot)
        
        # Video Thread
        self.worker1 = working1()
        self.worker1.start()
        self.worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        
    def VideoStopSlot(self):
        self.worker1.stop_thread()
        
    def VideoStartSlot(self):
        self.worker1.start_thread()
        
    def ImageUpdateSlot(self, Image):
         self.ui.LiveVideo.setPixmap(QPixmap.fromImage(Image))

    def returnCameraIndexes(self):
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr

class working1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    stopPlay = True 
    startPlay = False  
    def run(self): 
        self.ThreadActive = True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return
        while cap.isOpened():
            print(str(self.current_milli_time()))
            ret, img = cap.read()
            if not ret:
                break
            if (ret and self.stopPlay == True):
                if debug: cv2.imshow('image_rgb', img)

                # img = cv2.flip(img, 1)
                top, bottom, left, right = 75, 300, 75, 300  # bottom_left, bottom_right, bottom_left+image_size, bottom_right+image_size
                roi = img[top:bottom, left:right]
                #roi = cv2.flip(roi, 1)
                if debug: cv2.imshow('roi',roi)

                # converts image to gray and back to rgb so that gray image dimensions fit to model dimensions
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #gray = cv2.GaussianBlur(gray, (7, 7), 0)
                gray_3_dim = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                if debug: cv2.imshow('gray',gray_3_dim)

                alpha = self.classify(gray_3_dim)  # (roi)  # use (gray_3_dim) to use model to classify graysscale images or (roi) for rgb images

                # show rectangle with prediction area and predicted label on screen
                cv2.rectangle(img, (right, top), (left, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, alpha, (0, 430), font, 5, (0, 0, 255), 2)

                # show image in QT GUI
                #img = cv2.flip(img, 1)
                QtImg = QImage(img.data,img.shape[1],img.shape[0],QImage.Format_BGR888)
                pic = QtImg.scaled(450,450) #Qt Keep Aspect Ratio if needed
                self.ImageUpdate.emit(pic)
            else:    
                cap.release()
                cv2.destroyAllWindows()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break;

    def classify(self, image):
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        probs = model.predict(image)
        idx = np.argmax(probs)
        return alphabet[idx]

    def current_milli_time(self):
        return round(time.time() * 1000)

                        
    def stop_thread(self):
        self.stopPlay = False
        self.quit()
            
                  
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())            