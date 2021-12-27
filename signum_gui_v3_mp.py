import sys
import os
import glob
import time

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QPushButton
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from signum_gui_layout import Ui_MainWindow

import cv2
import mediapipe as mp
import numpy as np
from util import landmark_to_array, annotate_image
import pickle


#Pipeline for the camera
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
    ):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )


class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Start und Stop Button
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


class working1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    stopPlay = True 
    startPlay = False

    def run(self): 
        self.ThreadActive = True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            return

        model = self.load_latest_model()
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                #print(str(self.current_milli_time()))
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break  # continue
                if (success and self.stopPlay == True):
                    cv2.imshow('image_rgb', image)

                    # image = cv2.flip(image, 1)
                    top, bottom, left, right = 75, 375, 75, 375  # bottom_left, bottom_right, bottom_left+image_size, bottom_right+image_size
                    roi = image[top:bottom, left:right]
                    #roi = cv2.flip(roi, 1)
                    cv2.imshow('roi',roi)

                    # To improve performance, optionally mark the image as not writeable to pass by reference.
                    image.flags.writeable = False
                    results = hands.process(roi)

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    if results.multi_hand_landmarks:
                        annotate_image(roi, results)
                        # Extract landmarks from image (these could be passed to a classification algorithm)
                        landmarks = np.reshape(landmark_to_array(results), (1, -1))

                        prediction = model.predict(landmarks)
                        prediction_string = str(prediction[0])
                    else:
                        prediction_string = ' '

                    # show rectangle with prediction area and predicted label on screen
                    cv2.rectangle(image, (right, top), (left, bottom), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, prediction_string, (0, 430), font, 5, (0, 0, 255), 2)

                    # show image in QT GUI
                    #image = cv2.flip(image, 1)
                    QtImg = QImage(image.data,image.shape[1],image.shape[0],QImage.Format_BGR888)
                    pic = QtImg.scaled(450,450) #Qt Keep Aspect Ratio if needed
                    self.ImageUpdate.emit(pic)
                else:
                    cap.release()
                    cv2.destroyAllWindows()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break;

    def stop_thread(self):
        self.stopPlay = False
        self.quit()

    def current_milli_time(self):
        return round(time.time() * 1000)

    def load_latest_model(self):
        # File path containing saved models
        filepath = 'dataset/saved_model/'
        # necessary to load the latest saved model in the model folder
        list_of_files = glob.glob(filepath + '*.pkl')  # '*' means all if need specific format then e.g.: '*.h5'
        latest_file = max(list_of_files, key=os.path.getmtime)
        # head, tail = os.path.split(latest_file)
        # model_name = tail.split('.pkl')[0]

        # load trained model
        with open(latest_file, 'rb') as f:
            model = pickle.load(f)

        return model


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
