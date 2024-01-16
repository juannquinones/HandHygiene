import tkinter as tk 
from PIL import Image , ImageTk
import cv2
import imutils
from FuncionHigieneManos import *

ventana = tk.Tk()
ventana.geometry("510x502+200+50") # inicia en la posicion 200 + 10
ventana.title("Higiene Manos ")
ventana.resizable(width=False, height=False)
fondo = tk.PhotoImage(file="HIGIENE MANOS.png")
fondo1 = tk.Label(ventana, image=fondo).place(x=0, y=0, relheight=1, relwidth=1)

video = None

def HigieneManos(frame):
    global presente

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 2, ) as hands:
            
        #while video.isOpened():
            #ret, image = cap.read()
            #if not ret:
             #   print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
              #  break
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                #image.flags.writeable = True
        frame = cv2.flip(frame, -1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                #Draw the hand annotations on the image.
                #image.flags.writeable = False
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #height, width, _ = image.shape

        if results.multi_hand_landmarks:
            tiempo_inicio = time.time()*10

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=3))

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
            hand_language_class = model.predict(rows.reshape(1,-1))[0]
            prueba = hand_language_class
            cv2.rectangle(frame, (0,0), (250, 60), (245, 117, 16), -1)

            cv2.putText(frame, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(hand_language_class)
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                #     # Display Probability
            cv2.putText(frame, 'Time'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            end = time.time() *10
            presente=presente +(end-tiempo_inicio)
            #print(presente)
            cv2.putText(frame, str(round(presente,1))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            for i  in id_pasos:
                #print(i)
                if hand_language_class == i:
                    time_pasos[i-1]= time_pasos[i-1] +(end - tiempo_inicio)
            
                #cv2.putText(image, str(round(hand_language_prob[np.argmax(hand_language_prob)],2))
                #                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        #cv2.imshow('Prediccion', frame)

            #if cv2.waitKey(5) & 0xFF == 27:
             #   break
        #print('los tiempos son', time_pasos)
    return frame
def video_stream():
    global cap
    cap = cv2.VideoCapture(0)
    iniciar()

def iniciar():
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width= 458,height=590)
        frame = HigieneManos(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        etiqueta_video.configure(image=img)
        etiqueta_video.image = img
        etiqueta_video.after(10, iniciar)

def quitar():
    global video
    etiqueta_video.place_forget()
    video.release()


#botones 
boton = tk.Button(ventana, text="Empezar Higiene Manos", bg="#FF0000", relief="flat", cursor="hand2"
                    , command=video_stream,width=18, height=3, font=("Calisto MT", 12, "bold"))
boton.place(x=28, y=440)

boton2 = tk.Button(ventana, text="Guardar y Salir", bg="#FF0000", relief="flat", cursor="hand2"
                    , command=quitar,width=18, height=3, font=("Calisto MT", 12, "bold"))
boton2.place(x=299, y=440)



#Etiqueta 
etiqueta_video= tk.Label(ventana, bg="black")
etiqueta_video.place(x=32, y = 162)


ventana.mainloop()


