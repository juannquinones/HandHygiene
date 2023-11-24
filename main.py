import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Para ocultar los mensajes de advertencia de TensorFlow

class HandDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, 
                                         min_tracking_confidence=0.5, max_num_hands=2)

    def process_frame(self, image):
        return self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class HandDataProcessor:
    def process_hand_landmarks(self, results):
            if results.multi_hand_landmarks:
            # verificar primero la cantidad de landmarks
                if len(results.multi_hand_landmarks) == 1:
                    self.variable = results.multi_handedness[0].classification[0].label
                elif len(results.multi_hand_landmarks) == 2:
                    self.variable = 'Both'
                else:
                    self.variable = None
                    print('Error, se reconocen mas de dos manos')

            # Procesar la información de cada mano detectada
                if self.variable == "Left":
                    self.mano_dere_row = np.zeros(63)
                    self.mano_izq_row = np.array([[landmark.x, landmark.y, landmark.z] 
                                                for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
                elif self.variable == "Right":
                    self.mano_izq_row = np.zeros(63)
                    self.mano_dere_row = np.array([[landmark.x, landmark.y, landmark.z] 
                                                for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
                elif self.variable == "Both":
                    self.mano_dere_row = np.array([[landmark.x, landmark.y, landmark.z] 
                                                for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
                    self.mano_izq_row = np.array([[landmark.x, landmark.y, landmark.z] 
                                                for landmark in results.multi_hand_landmarks[1].landmark]).flatten()

            # Calcular centroides si hay datos válidos
                if self.mano_dere_row is not None and self.mano_izq_row is not None:
                    rows = np.concatenate((self.mano_dere_row, self.mano_izq_row))
                    rows = np.where(rows == 0, np.nan, rows)
                    self.centroid_x = np.nanmean(rows[::3])
                    self.centroid_y = np.nanmean(rows[1::3])
                    self.centroid_z = np.nanmean(rows[2::3])

                    rows = np.where(np.isnan(rows), 0, rows)
                    rows[::3] -= self.centroid_x
                    rows[1::3] -= self.centroid_y
                    rows[2::3] -= self.centroid_z

                    return rows
            return None

class Classifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def classify(self, rows):
        return self.model.predict(rows.reshape(1, -1))[0]

class UIRenderer:
    @staticmethod
    def draw_ui(self, image, results, info_str):
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibuja los landmarks y las conexiones entre ellos con colores y espesores específicos
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3)
                    )

        # Dibujar el rectángulo del fondo para el texto
            cv2.rectangle(image, (0, 0), (450, 60), (245, 117, 16), -1)
            
            # Dibujar el texto con información del tiempo y la clasificación
            cv2.putText(image, info_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Mostrar la imagen resultante
            cv2.imshow('Prediccion', image)
        # Aquí iría el código para dibujar en la imagen.

class Application:
    def __init__(self):
        self.detector = HandDetector()
        self.processor = HandDataProcessor()
        self.classifier = Classifier('avianca.pkl')
        self.ui_renderer = UIRenderer()
        self.cap = cv2.VideoCapture(0)
        self.run()

    def run(self):
        while self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            # Procesar imagen, detectar manos, clasificar y renderizar UI.
            # ...

        self.cap.release()
        cv2.destroyAllWindows()

# Ejecutar la aplicación
if __name__ == "__main__":
    app = Application()
