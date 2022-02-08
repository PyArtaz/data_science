import sys
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
from signum_v5_gui import Ui_MainWindow

import mediapipe as mp
from util import landmark_to_array, flip_coordinates, annotate_image, calc_dps
import bounding_box_mvp as bb
import predictor
jetson_nano_on = True

class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Buttons
        self.ui.w_stop.clicked.connect(self.start_stop_con)
        self.ui.textClear.clicked.connect(self.clear_text)

        # Checkboxes
        self.ui.letters.stateChanged.connect(self.collect_state)
        self.ui.numbers.stateChanged.connect(self.collect_state)
        self.ui.annotations.stateChanged.connect(self.collect_state)

        # Prediction Label
        self.ui.textBrowser.setReadOnly(False)

        # Initialize video Thread
        self.video_classifier = VideoClassification()
        self.video_classifier.start()

        # Connect custom signals to their respective functions
        self.video_classifier.image_update.connect(self.update_image)
        self.video_classifier.enter_text.connect(self.move_text)

    def start_stop_con(self):
        self.video_classifier.start_stop()

    def clear_text(self):
        self.ui.textBrowser.clear()
        self.ui.textBrowser_final.clear()

    def collect_state(self):
        self.video_classifier.collect_state()

    def update_image(self, image):
        self.ui.w_vid.setPixmap(QPixmap.fromImage(image))

    def move_text(self, text):
        self.ui.textBrowser.clear()
        self.ui.textBrowser_final.insertPlainText(text)
        self.ui.textBrowser_final.insertPlainText('\n')

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

class VideoClassification(QThread):
    image_update = pyqtSignal(QImage)
    enter_text = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.stopPlay = True
        self.analyzing = True
        self.letters = True
        self.numbers = True
        self.annotations = True

        # Initialize Predictors
        self.ln_predictor = predictor.Predictor('./models/model_Own_landmarks_bb_squarePix_Letters+Digits.sav', 20)
        self.letters_predictor = predictor.Predictor('./models/model_Own_landmarks_bb_squarePix_Letters.sav', 20)
        self.numbers_predictor = predictor.Predictor('./models/model_Own_landmarks_bb_squarePix_Digits_with_enter-space-del.sav', 20)

    def run(self):
        self.ThreadActive = True
	
        if(jetson_nano_on):
            cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened(): 
            print("Cant access Camera")
            return

        data_logger = predictor.DataLogger(limit=2)
        mp_hands = mp.solutions.hands
        fps_start = time.time()

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                if ret & self.stopPlay:
                    # Flip image and pass RGB image to mediapipe
                    image = cv2.flip(image, 1)
                    image.flags.writeable = False
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    image.flags.writeable = True

                    # Check if a hand is detected
                    if results.multi_hand_landmarks:
                        # Extract landmarks from results object
                        landmarks = landmark_to_array(results).reshape(1, -1)
                        data_logger.hand_detected()

                        # Create bounding box, make it square and normalize the coordinates
                        box = bb.BoundingBox(landmarks, image.shape[1], image.shape[0])
                        box.make_square()
                        landmarks_bb = box.coordinates_to_bb(landmarks)

                        # Check for left hand and flip coordinates if left hand is detected
                        hand = results.multi_handedness[-1].classification[0].label
                        if hand == 'Left':
                            landmarks_bb = flip_coordinates(landmarks_bb)

                        if self.letters or self.numbers:
                            # Generate prediction from landmarks and display it in UI
                            prediction, probability_dict = self.handle_predictions(landmarks_bb)
                            data_logger.add_prediction(prediction)

                            prediction_string = prediction + ' - ' + str(probability_dict.get(prediction)) + '%'
                            win.ui.current_prediction.setText(prediction_string)
                        else:
                            win.ui.current_prediction.clear()

                        # Annotate the image with both, a bounding box and the landmark positions
                        if self.annotations:
                            image = box.draw(image)
                            annotate_image(image, results, jetson_nano_on)

                    else:
                        win.ui.current_prediction.clear()

                    # Set status when predictions are logged
                    if data_logger.time < time.time():
                        self.analyzing = True
                        self.prediction_status()

                    # Observe the final prediction
                    if data_logger.timeout():
                        self.analyzing = False
                        if self.prediction_status(data_logger.final_prediction()):
                            self.functionality_mapper(prediction)

                    dps, fps_start = calc_dps(fps_start)
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, 'FPS: {:.2f}'.format(dps), (5, 30), font, 1, (0, 255, 255), 1)
                    # Show image in UI
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    qt_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    pic = qt_img.scaled(1280, 720, Qt.KeepAspectRatio)
                    self.image_update.emit(pic)
                    if not self.stopPlay:
                        self.quit()
                        win.ui.w_stop.setText('Start')
                else:
                    cap.release()
                    cv2.destroyAllWindows()

    def handle_predictions(self, landmarks):
        """
        Update the prediction by classifying the current landmarks. By default, a model for numbers and letters is used,
        but optionally a model only trained for letters or numbers can be used, respectively.

        Parameters
        ----------
        landmarks : ndarray
            The landmarks to be predicted. Shape 1 x 63.

        Returns
        -------

        """
        if self.letters and self.numbers:
            self.ln_predictor.update_prediction(landmarks)
            prediction = self.ln_predictor.get_prediction()
            probability_dict = self.ln_predictor.create_probability_dict()

        if self.letters and not self.numbers:
            self.letters_predictor.update_prediction(landmarks)
            prediction = self.letters_predictor.get_prediction()
            probability_dict = self.letters_predictor.create_probability_dict()
        elif self.numbers and not self.letters:
            self.numbers_predictor.update_prediction(landmarks)
            prediction = self.numbers_predictor.get_prediction()
            probability_dict = self.numbers_predictor.create_probability_dict()

        return prediction, probability_dict

    def prediction_status(self, prediction=None):
        """
        Write the status of the prediction to the UI. Then, indicate whether a prediction could be made or not.

        Parameters
        ----------
        prediction : str
            The final prediction made by the DataLogger object.

        Returns
        -------
        bool
            Whether a prediction could be made or not.

        """
        if self.analyzing:
            win.ui.status.setText('Analyzing...')
        else:
            if prediction is None:
                win.ui.status.setText('Detection failed, retrying!')
            elif prediction == '-1':
                win.ui.status.setText('No hand detected, retrying!')
            elif prediction == '-2':
                win.ui.status.setText('Insufficient data, retrying!')
            else:
                win.ui.status.setText('Success!')
                return True

        return False

    def functionality_mapper(self, prediction):
        """
        Maps the predictions Space, Del and Enter to their respective functions on the Qt textBox.

        Parameters
        ----------
        prediction : str
            The final prediction made by the DataLogger object.

        Returns
        -------

        """
        if prediction == 'SPACE':
            win.ui.textBrowser.insertPlainText(' ')
        elif prediction == 'DEL':
            win.ui.textBrowser.textCursor().deletePreviousChar()
        elif prediction == 'ENTER':
            self.enter_text.emit(win.ui.textBrowser.toPlainText())
        else:
            win.ui.textBrowser.insertPlainText(prediction)

    def start_stop(self):
        """
        Pause or restart the video classification thread on click of the respective button.

        Parameters
        ----------

        Returns
        -------

        """
        self.stopPlay = not self.stopPlay
        if self.stopPlay:
            self.start()
            win.ui.w_stop.setText('Pause')

    def collect_state(self):
        """
        Change values of class variables when the state of a checkbox changes.

        Parameters
        ----------

        Returns
        -------

        """
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
