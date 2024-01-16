
import logging
from token import EXACT_TOKEN_TYPES
from tracemalloc import start
from sklearn.metrics import accuracy_score # Accuracy metrics 
import cv2
import csv
import mediapipe as mp
import numpy as np
import pandas as pd 
import os 
import pickle 
import time 
import threading

with open(r'C:\Users\Andres Romero\Documents\Proyectos\hands_detection\scripts\RF05062023.pkl', 'rb') as f:
     model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
variable =None
tiempo_inicio = 0
presente =0
contador =0
pasos = np.array([False, False, False, False, False, False])
id_pasos = [1,2,3,4,5,6]
time_pasos =[0,0,0,0,0,0]
#cap = cv2.VideoCapture(0) se cambio en pruebainter cap por video y image por frame
#cap = cv2.VideoCapture(0)


        #cap.release()
        #scv2.destroyAllWindows()
        
#while True:
#HigieneManos()
