import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from HandHygieneMain import *
import mediapipe as mp
import time
import pickle
import os
import sys

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, model_path):
        super().__init__()
        self.restart_settings(model_path)
    
    def restart_settings(self, model_path):
        self.video = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video = cv2.VideoWriter(self.filename, fourcc, 40.0, (frame_width, frame_height))


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
                self.video.write(image)
        finally:
            self.cap.release()


    def save_recording(self):
        if self.video is not None:
            self.video.release() 
            self.video = None  # Reinicia el objeto para permitir nuevas grabaciones
            print(f"Video guardado como {self.filename}")
        else:
            print("No hay grabación activa.")
