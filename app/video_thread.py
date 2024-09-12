import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from HandHygieneMain import *
import mediapipe as mp

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.restart_settings()
    
    def restart_settings(self):
        self.cap = None
        self._run_flag = False
        self._source = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands = 2,static_image_mode=True) # modelo
        self.image_success = True
        self.hand_model = HandHygineModel(self.mp_drawing, self.mp_drawing_styles, self.mp_hands, self.hands)#, step_prediction_model=self.modelo)

    def set_source(self, source):
        self._source = source

    def run(self):
        self._run_flag = True
        self.cap = cv2.VideoCapture(self._source)
        self.cap.set(cv2.CAP_PROP_FPS, 40)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while self._run_flag:
                image_success, image = self.cap.read()
                success, image2, right_hand_rows, left_hand_rows = self.hand_model.get_landmarks_structure(success=image_success, image = image, mode='capture', return_image=True)
                if image_success:
                    self.change_pixmap_signal.emit(image)
                else:
                    break
        finally:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
    
    def restart(self):
        self.restart_settings()
