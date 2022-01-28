import sys
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QPushButton
)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from signum_v5_gui import Ui_MainWindow
import time
import cv2
import mediapipe as mp
import numpy as np
from util_mvp_v1 import landmark_to_array, flip_coordinates, mode, annotate_image, calc_dps, prediction_checker, functionality_mapper
import bounding_box_mvp as bb
import predictor_v1
jetson_nano_on = False

class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(QMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Start und Stop Button
        self.ui.w_stop.clicked.connect(self.VideoStopSlot)

        # Checkboxes
        self.ui.letters.stateChanged.connect(self.collect_state)
        self.ui.numbers.stateChanged.connect(self.collect_state)
        self.ui.annotations.stateChanged.connect(self.collect_state)

        # Prediction Label
        self.ui.textBrowser.setReadOnly(False)

        # Video Thread
        self.worker1 = working1()
        self.worker1.start()
        self.worker1.ImageUpdate.connect(self.ImageUpdateSlot)

    def VideoStopSlot(self):
        self.worker1.stop_thread()

    def collect_state(self, state):
        self.worker1.collect_state(state)

    def ImageUpdateSlot(self, Image):
        self.ui.w_vid.setPixmap(QPixmap.fromImage(Image))

if(jetson_nano_on):
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

class working1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    stopPlay = True
    letters = True
    numbers = True
    annotations = True

    def run(self):
        self.ThreadActive = True
	
        if(jetson_nano_on):
            cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened(): 
            print("Cant access Camera")
            return

        # Initialize Predictors
        ln_predictor = predictor_v1.Predictor('./models/model_Own_landmarks_bb_squarePix.sav', 20)
        letters_predictor = predictor_v1.Predictor('./models/model_Letters.sav', 20)
        numbers_predictor = predictor_v1.Predictor('./models/model_digits_without_unknowns.sav', 20)

        data_logger = predictor_v1.DataLogger(limit=2)


        mp_hands = mp.solutions.hands

        fps_start = time.time()

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret: break
                if ret & self.stopPlay:
                    image = cv2.flip(image, 1)

                    image.flags.writeable = False
                    # Converting to RGB drastically improved mediapipe tracking robustness
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    image.flags.writeable = True

                    if results.multi_hand_landmarks:
                        # Extract landmarks from image
                        landmarks = landmark_to_array(results).reshape(1, -1)
                        data_logger.hand_detected()
                        # Create bounding box and make it square
                        box = bb.BoundingBox(landmarks, image.shape[1], image.shape[0])
                        box.make_square()

                        # Coordinate transformation
                        landmarks_bb = box.coordinates_to_bb(landmarks)

                        # Check for left hand and flip coordinates if left hand is detected
                        hand = results.multi_handedness[-1].classification[0].label
                        if hand == 'Left':
                            landmarks_bb = flip_coordinates(landmarks_bb)

                        # Update the predictions by classifying the current landmarks
                        # By default, a model for numbers and letters is used, but optionally a model only trained for
                        # letters or numbers can be used, respectively.
                        if self.letters or self.numbers:
                            if self.letters and self.numbers:
                                ln_predictor.update_prediction(landmarks_bb)
                                prediction, prob_prediction = ln_predictor.get_all_predictions()
                                probability_dict = ln_predictor.create_probability_dict()

                            if self.letters and not self.numbers:
                                letters_predictor.update_prediction(landmarks_bb)
                                prediction, prob_prediction = letters_predictor.get_all_predictions()
                                probability_dict = letters_predictor.create_probability_dict()
                            elif self.numbers and not self.letters:
                                numbers_predictor.update_prediction(landmarks_bb)
                                prediction, prob_prediction = numbers_predictor.get_all_predictions()
                                probability_dict = numbers_predictor.create_probability_dict()

                            data_logger.add_prediction(prediction)

                            # Assemble the prediction string (1. with mode, 2. with prediction from mean probability)
                            # prediction_string = str(prediction) + '-' + str(probability_dict.get(prediction))
                            prediction_string = prediction + '/' + prob_prediction + '-' + str(probability_dict.get(prediction))

                        # Annotate the image with both, a bounding box and the landmark positions
                        if self.annotations:
                            image = box.draw(image)
                            annotate_image(image, results)

                    else:
                        prediction_string = ' '

                    # Progress bar code (not working yet)
                    #progress, steps = data_logger.progress()
                    #win.ui.progressBar.setMaximum(steps)
                    #win.ui.progressBar.setValue(progress)

                    # Observe the final prediction
                    if data_logger.timeout():
                        prediction = data_logger.final_prediction()
                        if prediction is None:
                            win.ui.status.setText('Detection failed, retrying!')
                        elif prediction == '-1':
                            win.ui.status.setText('No hand detected, retrying!')
                        elif prediction == '-2':
                            win.ui.status.setText('Insufficient data, retrying!')
                        else:
                            # Map the final prediction to its function
                            functionality_mapper(prediction, win.ui.textBrowser)
                            if prediction == 'ENTER':
                                win.ui.textBrowser_final.insertPlainText(win.ui.textBrowser.toPlainText())
                            win.ui.status.setText('Success!')

                    # Set status if the predictions are logged
                    if data_logger.time < time.time():
                        win.ui.status.setText('Analyzing...')

                    # Write current prediction to Qlabel
                    win.ui.current_prediction.setText(prediction_string)    
                    dps, fps_start = calc_dps(fps_start)
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, 'FPS: {:.2f}'.format(dps), (5, 30), font, 1, (0, 255, 255), 1)    
                    # show image in QT GUI
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    QtImg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    pic = QtImg.scaled(1280, 720, Qt.KeepAspectRatio)  # Qt Keep Aspect Ratio if needed
                    self.ImageUpdate.emit(pic)

                else:
                    cap.release()
                    cv2.destroyAllWindows()

    def stop_thread(self):
        self.stopPlay = False
        self.quit()

    # Change values of class variables when the state of a checkbox changes
    def collect_state(self, state):
        if win.ui.letters.isChecked():
            self.letters = True
        elif not win.ui.letters.isChecked():
            self.letters = False

        if win.ui.numbers.isChecked():
            self.numbers = True
        elif not win.ui.numbers.isChecked():
            self.numbers = False

        if win.ui.annotations.isChecked():
            self.annotations = True
        elif not win.ui.annotations.isChecked():
            self.annotations = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
