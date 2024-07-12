import mediapipe as mp
import os
import sys
import warnings
from tkinter import filedialog
from tkinter import messagebox
from tkinter import StringVar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HandHygiene')))
print("sys.path:", sys.path)
from HandHygieneMain import *
# MODEL_PATH = r'D:\\Proyectos\\Hands\\HigieneManos\\Data\\Models\\rf_5es_98acc.pkl'

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import random
import pickle
import numpy as np
from collections import deque
import time


# Suprimir warnings
warnings.filterwarnings("ignore")

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualización de cámara y clasificación de pasos de higiene")
        self.root.geometry("800x600")
        
        # Inicialización de la cámara y modelo
        self.cap = None
        self.model_initialized = False
        self.modelo = None
        self.load_model()
        
        # Deques y variables para el procesamiento del video
        self.step_id = [1, 2, 3, 4, 5, 6, 'No Step']
        self.step_time = [0] * len(self.step_id)
        self.step_time_deque = deque(maxlen=2)
        self.video_start_time = time.time()
        self.last_prediction = None
        self.processing_video = False  # Bandera para controlar el procesamiento del video
        self.start_time = None  # Variable para almacenar el tiempo de inicio
        
        # Elementos de la interfaz tkinter
        self.label = tk.Label(self.root)
        self.label.pack(padx=10, pady=10)
        
        self.btn_start_realtime = tk.Button(self.root, text="Iniciar Cámara en Tiempo Real", command=self.start_realtime)
        self.btn_start_realtime.pack(pady=10)
        
        self.btn_load_video = tk.Button(self.root, text="Cargar Video", command=self.load_video)
        self.btn_load_video.pack(pady=10)
        
        self.btn_main_menu = tk.Button(self.root, text="Volver al Menú Principal", command=self.main_menu)
        self.btn_main_menu.pack(pady=10)
        
        self.btn_restart = tk.Button(self.root, text="Reiniciar Proceso", command=self.restart_process)
        self.btn_restart.pack(pady=10)
        
        self.time_var = StringVar()
        self.time_label = tk.Label(self.root, textvariable=self.time_var)
        self.time_label.pack(pady=10)
        
        self.step_time_deque.append(time.time())
        self.update_buttons(False)

    # Método para cargar el modelo de clasificación de pasos de higiene
    def load_model(self):
        MODEL_PATH = '/Users/juannquinones/Library/CloudStorage/OneDrive-ESCUELACOLOMBIANADEINGENIERIAJULIOGARAVITO/Nico/Manos/HigieneManos/Data/Models/rf_260624.pkl'
        with open(MODEL_PATH, 'rb') as file:
            self.modelo = pickle.load(file)
        self.model_initialized = True
    
    # Método para iniciar la cámara en tiempo real
    def start_realtime(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.processing_video = True
        self.start_time = time.time()
        self.update_buttons(True)
        self.process_video()
    
    # Método para cargar un video
    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)  # Iniciar la captura desde el video seleccionado
            self.processing_video = True
            self.start_time = time.time()
            self.update_buttons(True)
            self.process_video()
    
    # Método para detener el procesamiento de video
    def stop_process(self):
        self.processing_video = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.update_buttons(False)
    def display_last_frame(self):
        if self.last_image is not None:
            self.display_frame(self.last_image)
        
    
    # Método para volver al menú principal
    def main_menu(self):
        self.stop_process()
        self.label.config(image='')
        self.time_var.set('')
        self.step_time = [0] * len(self.step_id)
        self.last_prediction = None
        messagebox.showinfo("Menú Principal", "Has vuelto al menú principal.")

    # Método para reiniciar el proceso actual
    def restart_process(self):
        self.stop_process()
        self.start_time = time.time()
        self.step_time_deque.append(time.time())
        self.last_prediction = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'video_path') and self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.processing_video = True
            self.process_video()
        else:
            self.start_realtime()
    
    # Método para procesar el video y realizar la clasificación de pasos de higiene
    def process_video(self):
        if self.cap is not None and self.cap.isOpened() and self.processing_video:
            success, image = self.cap.read()
            if success:
                image = self.classify_hygiene_step(image)
                self.last_image = image  # Guardar la última imagen procesada
                self.display_frame(image)
                self.update_time()
                self.root.after(10, self.process_video)
            else:
                            # Se llegó al final del video
                self.stop_process()  # Detener el procesamiento del video
                self.display_last_frame()  # Mostrar la última imagen procesada
                print("Parece que este es el fin del video")
        else:
            print("Error: La cámara o el video no están disponibles.")
    
    # Método para clasificar el paso de higiene en la imagen dada
    def classify_hygiene_step(self, image):
        if self.model_initialized:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_hands = mp.solutions.hands
            # Establecer max_num_hands a 2 para usar la CPU en lugar de la GPU
            hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands=2, static_image_mode=True)
            hand_model = HandHygineModel(mp_drawing, mp_drawing_styles, mp_hands, hands, step_prediction_model=self.modelo)
            
            success, _, right_hand_rows, left_hand_rows = hand_model.get_landmarks_structure(success=True, image=image, mode='capture', return_image=True)
            
            if success and hand_model.verify_hand_rows(right_hand_rows, left_hand_rows):
                X = np.concatenate([hand_model.get_normalized_rows(right_hand_rows), hand_model.get_normalized_rows(left_hand_rows)], axis=0).reshape(42*3)
                y = hand_model.predict_hygiene_step(X.reshape(1,-1))

                if y != self.last_prediction:
                    end_time = time.time()
                    duration = end_time - self.video_start_time
                    if self.last_prediction is None:
                        self.step_time[-1] += duration
                    else:
                        self.step_time[self.last_prediction]+= duration
                    self.last_prediction =y
                    self.video_start_time = end_time
                
                cv2.putText(image, f"Step: {y}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
        
        return image
    
    # Método para mostrar la imagen procesada en la interfaz tkinter
    def display_frame(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        img = Image.fromarray(image)  # Convertir a formato Image
        img = ImageTk.PhotoImage(image=img)  # Convertir a formato compatible con tkinter
        self.label.img = img
        self.label.config(image=img)
    
    # Método para actualizar el estado de los botones
    def update_buttons(self, processing):
        if processing:
            self.btn_start_realtime.pack_forget()
            self.btn_load_video.pack_forget()
            self.btn_main_menu.pack()
            self.btn_restart.pack()
        else:
            self.btn_main_menu.pack_forget()
            self.btn_restart.pack_forget()
            self.btn_start_realtime.pack()
            self.btn_load_video.pack()
    
    # Método para actualizar el tiempo transcurrido
    def update_time(self):
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            self.time_var.set(f"Tiempo transcurrido: {elapsed_time:.2f} segundos")

# Crear la ventana principal de tkinter y ejecutar la aplicación
root = tk.Tk()
app = CameraApp(root)
root.mainloop()
