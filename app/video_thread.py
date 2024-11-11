import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from HandHygieneMain import *
import mediapipe as mp
from datetime import datetime
import time
import pickle
import os
import sys

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, model_path, app_path):
        super().__init__()
        self.steps_map = {1:'1', 2:'2.1', 3:'2.2', 4:'3', 5:'4.1', 
                         6:'4.2', 7:'5.1', 8:'5.2', 9:'6.1', 10:'6.2', 11:'No step'}
        self.app_db = os.path.join(app_path, 'DataBase')
        self.temp_filename = os.path.join(self.app_db, 'videoSalida.mp4')
        self.cap = None
        self.video = None
        self.restart_settings(model_path)
        
    def restart_settings(self, model_path):
        if self.cap is not None:
            self.cap.release()
            self.video.release()
            self.cap = None
            self.video = None
        self._run_flag = False
        self._source = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands = 2,static_image_mode=True) # modelo
        self.image_success = True
        #video recording
        # self.video = cv2.VideoWriter(self.temp_filename,cv2.VideoWriter_fourcc(*'mp4v'),40.0,(640,480))

        #MODEL_PATH ='/Users/juannquinones/Library/CloudStorage/OneDrive-ESCUELACOLOMBIANADEINGENIERIAJULIOGARAVITO/Nico/Manos/HigieneManos/Data/Models/rf_260624.pkl'
        MODEL_PATH = model_path
        with open(MODEL_PATH, 'rb') as file:
            self.model = pickle.load(file)

        self.hand_model = HandHygineModel(self.mp_drawing, self.mp_drawing_styles, self.mp_hands, self.hands, self.model)#, step_prediction_model=self.modelo)

        # Variables de la clase dinamicas
        self.step_time =[0,0,0,0,0,0,0,0,0,0,0]
        self.video_start_time = time.time()
        self.last_prediction = 11
        self.y=None

    def set_source(self, source):
        self._source = source

    def run(self):
        self._run_flag = True
        self.cap = cv2.VideoCapture(self._source)
        self.cap.set(cv2.CAP_PROP_FPS, 40)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video = cv2.VideoWriter(self.temp_filename,cv2.VideoWriter_fourcc(*'H264'),40.0,(frame_width,frame_height))
        try:
            while self._run_flag:
                image_success, image = self.cap.read()
                success, _, right_hand_rows, left_hand_rows = self.hand_model.get_landmarks_structure(success=image_success, image = image, mode='capture', return_image=True)
                # if image_success:
                #     self.change_pixmap_signal.emit(image)
                self.video.write(image)
                if success: # Solo se procesan las que tienen landmarks validos 
                    
                    if self.hand_model.verify_hand_rows(right_hand_rows,left_hand_rows):
                        X = np.concatenate([self.hand_model.get_normalized_rows(right_hand_rows), self.hand_model.get_normalized_rows(left_hand_rows)], axis=0).reshape(42*3) 
                        self.y = self.hand_model.predict_hygiene_step(X.reshape(1,-1))+1

                        if self.y != self.last_prediction:
                            end_time = time.time()
                            duration = end_time - self.video_start_time
                            self.step_time[self.last_prediction-1]+= duration
                            self.last_prediction = self.y
                            self.video_start_time = end_time
                    cv2.putText(image, f"Step: {self.steps_map[self.y]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
                    self.change_pixmap_signal.emit(image)
                    #cv2.imshow('Hand step clasiffication in Real Time', image)
        finally:
            self.video.release() 
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def get_steps_times(self):
        return self.step_time
    
    def restart(self):
        self.restart_settings()

    def save_recording(self, path, id):
        if self.video is not None and id is not None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            final_filename = os.path.join(self.app_db, f'recording_{id}_{timestamp}.mp4')
            os.rename(self.temp_filename, final_filename)
            self.video.release() 
            self.cap.release()
            self.video = None
            self.cap = None
