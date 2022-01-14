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
from util_mvp import landmark_to_array, flip_coordinates, mode, annotate_image
import bounding_box_mvp as bb


class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(QMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Start und Stop Button
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
        if not cap.isOpened():
            return

        mp_hands = mp.solutions.hands

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret: break
                if ret & self.stopPlay:
                    image = cv2.flip(image, 1)

                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    image.flags.writeable = True

                    if results.multi_hand_landmarks:
                        # Extract landmarks from image
                        landmarks = landmark_to_array(results).reshape(1, -1)

                        # Create bounding box and make it square
                        box = bb.BoundingBox(landmarks, image.shape[1], image.shape[0])
                        box.make_square()

                        # Coordinate transformation
                        landmarks_bb = box.coordinates_to_bb(landmarks)

                        # Check for left hand and flip coordinates if left hand is detected
                        hand = results.multi_handedness[-1].classification[0].label
                        if hand == 'Left':
                            landmarks_bb = flip_coordinates(landmarks_bb)

                        # Annotate the image
                        bb_image = box.draw(image)
                        annotate_image(bb_image, results)

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # show image in QT GUI
                    QtImg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    pic = QtImg.scaled(1280, 720, Qt.KeepAspectRatio)  # Qt Keep Aspect Ratio if needed
                    self.ImageUpdate.emit(pic)

                else:
                    cap.release()
                    cv2.destroyAllWindows()

    def stop_thread(self):
        self.stopPlay = False
        self.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())