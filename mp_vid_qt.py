import sys
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QPushButton
)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from signum_v4_gui import Ui_MainWindow

import cv2
import mediapipe as mp
import numpy as np
from util_mvp import landmark_to_array, annotate_image


class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
            super(QMainWindow, self).__init__()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

            
            #Start und Stop Button
            self.ui.w_stop.clicked.connect(self.VideoStopSlot)
            
            # Video Thread
            self.worker1 = working1()
            self.worker1.start()
            self.worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        
    def VideoStopSlot(self):
        self.worker1.stop_thread()
        
    def ImageUpdateSlot(self, Image):
         self.ui.w_vid.setPixmap(QPixmap.fromImage(Image))
         

class working1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    stopPlay = True 
      
    def run(self): 
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return
        
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():				    
                ret, image = cap.read()
                if not ret: break
                if (ret & self.stopPlay == True):
                    
                    top, bottom, left, right = 75, 275, 375, 575  # bottom_left, bottom_right, bottom_left+image_size, bottom_right+image_size
                    roi = image[top:bottom, left:right]
                    
                    results = hands.process(roi)
                    image.flags.writeable = True
                    
                    if results.multi_hand_landmarks:
                        annotate_image(roi, results)
                        # Extract landmarks from image (these could be passed to a classification algorithm)
                        landmarks = np.reshape(landmark_to_array(results), (1, -1))

                    # show rectangle with prediction area and predicted label on screen
    
                    cv2.rectangle(image, (right, top), (left, bottom), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # show image in QT GUI
                    image = cv2.flip(image, 1)
                    QtImg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    pic = QtImg.scaled(1280,720,Qt.KeepAspectRatio)  # Qt Keep Aspect Ratio if needed
                    self.ImageUpdate.emit(pic)
                    
                else:    
                    cap.release()
                    cv2.destroyAllWindows()
                            
    def stop_thread(self):
        self.stopPlay = False
        self.quit()
        
# Defines with which probability the detected landmarks are printed to the console
#print_probability = 0.0025
# Specify whether video should be saved or not
#save_video = False

#rng = np.random.default_rng()
#mp_hands = mp.solutions.hands

# For webcam input:
#cap = cv2.VideoCapture(0)
#if save_video:
#    frame_width = int(cap.get(3))
#    frame_height = int(cap.get(4))
#    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (frame_width, frame_height))

#with mp_hands.Hands(
#        min_detection_confidence=0.5,
#        min_tracking_confidence=0.5) as hands:
#    while cap.isOpened():
#        success, image = cap.read()
#        if not success:
#            print("Ignoring empty camera frame.")
#            # If loading a video, use 'break' instead of 'continue'.
#            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
#        image.flags.writeable = False
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        results = hands.process(image)

        # Draw the hand annotations on the image.
#        image.flags.writeable = True
#        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#        if results.multi_hand_landmarks:
#            annotate_image(image, results)
#            # Extract landmarks from image (these could be passed to a classification algorithm)
#            landmarks = np.reshape(landmark_to_array(results), (1, -1))
#            if rng.random() < print_probability:
#                print(landmarks)

#        if save_video:
#            out.write(image)
#        # Flip the image horizontally for a selfie-view display.
#        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#        if cv2.waitKey(5) & 0xFF == 27:
#            break
#cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_()) 