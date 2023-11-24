
import cv2
import mediapipe as mp
import numpy as np 
import pickle 
import time 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Para ocultar los mensajes de advertencia de TensorFlow

#import tensorflow as tf
# Configurar el uso de la GPU
#gpus = tf.config.experimental.list_physical_devices('GPU')

with open('modelo.pkl', 'rb') as f:
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
time_pasos = np.zeros(6)

#time_pasos =[0,0,0,0,0,0]
#cap = cv2.VideoCapture("Manos_Completo.mp4")

cap = cv2.VideoCapture(0)


with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 2, ) as hands:
        
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            #image.flags.writeable = True
        image = cv2.flip(image, -1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            #Draw the hand annotations on the image.
            #image.flags.writeable = False
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

        if results.multi_hand_landmarks:
            tiempo_inicio = time.time()*10
            custom_point_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            #custom_point_style.color = (0, 0, 255)  # Color en formato BGR (azul)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image,hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=3))
                #mp_drawing.draw_landmarks(image, mp.solutions.drawing_utils.DRAW_SPECIFIC_POINTS,
                 #       {'landmark_points': [(mp.solutions.hands.HandLandmark.WRIST, (blue_point))]})

                # verificar primero la cantidad de landmarks
            if len(results.multi_hand_landmarks) == 1:
                variable =results.multi_handedness[0].classification[0].label
            elif len(results.multi_hand_landmarks) == 2:
                variable = 'Both'
            else:
                variable = None

                #try:
            if variable =="Left":
                mano_dere_row = np.zeros(63)
                mano_izq_row = np.array([[landmark.x, landmark.y,landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
            elif variable == "Right":
                mano_izq_row = np.zeros(63)
                mano_dere_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
            elif variable == "Both":
                mano_dere_row = np.array([[landmark.x, landmark.y,landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
                mano_izq_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[1].landmark]).flatten()
            else: 
                print('Error, se reconocen mas de dos manos')
                
            rows = np.concatenate((mano_dere_row,mano_izq_row))
            rows = np.where(rows ==0, np.nan,rows)
            centroid_x = np.nanmean(rows[::3])
            centroid_y = np.nanmean(rows[1::3])
            centroid_z = np.nanmean(rows[2::3])
            blue_point = (centroid_x, centroid_y, centroid_z)
            
            rows = np.where(np.isnan(rows), 0, rows)
            rows[::3] -= centroid_x
            rows[1::3] -= centroid_y
            rows[2::3] -= centroid_z

            hand_language_class = model.predict(rows.reshape(1,-1))[0]
            prueba = hand_language_class
            cv2.rectangle(image, (0,0), (450, 60), (245, 117, 16), -1)
            #cv2.circle(image, blue_point, radius=25, color=(255, 0, 0), thickness=-1)  # El color (255, 0, 0) representa el azul

                #     # Display Probability
            end = time.time() *10
            presente=presente +(end-tiempo_inicio)
            #print(presente)
            info_str = 'TIME: {:.1f} | CLASS: {}'.format(round(presente,1), hand_language_class)
            cv2.putText(image, info_str, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            for i, id_paso in enumerate(id_pasos):
                if hand_language_class == id_paso:
                    time_pasos[i] += end - tiempo_inicio

            #for i  in id_pasos:
                #print(i)
                #if hand_language_class == i:
                    #time_pasos[i-1]= time_pasos[i-1] +(end - tiempo_inicio)
            
                #cv2.putText(image, str(round(hand_language_prob[np.argmax(hand_language_prob)],2))
                 #                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Prediccion', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    print('los tiempos son', time_pasos)
    cap.release()
    cv2.destroyAllWindows()
    

