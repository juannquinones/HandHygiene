import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self._source = 0

    def set_source(self, source):
        self._source = source

    def run(self):
        self._run_flag = True
        cap = cv2.VideoCapture(self._source)
        try:
            while self._run_flag:
                ret, cv_img = cap.read()
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
                else:
                    break
        finally:
            cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
