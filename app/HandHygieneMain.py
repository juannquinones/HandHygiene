import os
import re
import numpy as np
import cv2
from collections import Counter, deque


class HandHygineModel:
    def __init__(self,mp_drawing, mp_drawing_styles, mp_hands, hands, step_prediction_model=None):
        self.frames_prediction =  deque(maxlen=15) 
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.mp_hands = mp_hands
        self.hands = hands
        self.step_prediction_model = step_prediction_model

    def get_landmarks_structure(self, success, image, mode, return_image=True):
        '''
        This function read each frame in a video and returns the landmarks in a array
        Parameters
        ----------
        succes: bool
            Boolean indicating the success of reading a frame.
        image: numpy.ndarray
            The image frame.
        mode: str
            Indicating the mode, either 'video' to load a video or 'capture' when the camera is capturing video.
        return_image: bool
            Indicating wheter or not  show the video.
            
        Returns
        ----------
        success: bool
            Indicating the success of reading a frame
        image: numpu.ndarray
            The image frame
        right_hand_rows: numpy.ndarray or none 
            Array containing the landmarks of the right hand if detected, otherwise None.
        left_hand_rows: numpy.nd.array or none
            Array containing the landmarks of the left hand if detected, otherwise None.
        '''
        if not success: # validate if success
            if mode == 'video': # if not success and is a video, stop
                #print("Empty camera frame")
                return False, None, None, None 
            elif mode == 'capture': # if not succes and the camera is capturing wait for the next image
                #print("Ignoring empty camera frame.")
                pass
            else:
                #print('Select a valid mode')
                return False, None, None, None
        if image is None:
            print('Error while reading images ')
            return False, None, None, None
        else: 

            results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            #height, width, _ = image.shape
            if results.multi_hand_landmarks:
                if return_image:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image,hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                        #self.mp_self.hands.HAND_CONNECTIONS,self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=3))

                # verificar primero la cantidad de landmarks
                if len(results.multi_hand_landmarks) == 1:
                    variable =results.multi_handedness[0].classification[0].label
                elif len(results.multi_hand_landmarks) == 2:
                    variable = 'Both'
                else:
                    variable = None
                
                if variable =="Left":
                    right_hand_rows = np.zeros([21,3])
                    left_hand_rows = np.array([[landmark.x, landmark.y,landmark.z] for landmark in results.multi_hand_landmarks[0].landmark])
                elif variable == "Right":
                    left_hand_rows = np.zeros([21,3])
                    right_hand_rows = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark])
                elif variable == "Both":
                    right_hand_rows = np.array([[landmark.x, landmark.y,landmark.z] for landmark in results.multi_hand_landmarks[0].landmark])
                    left_hand_rows = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[1].landmark])
                else:
                    print('Error, se reconocen mas de dos manos')
                    return False, None, None, None
                return True, image, right_hand_rows, left_hand_rows 
            return False, None, None, None

    def predict_hygiene_step(self, normalized_points):
        if self.step_prediction_model is None:
            raise Exception('There is no step_prediction_model. Please set the model using set_model function')
        else:
            #pred = self.step_prediction_model.predict(normalized_points.reshape(1,-1))[0]
            class_probabilities = self.step_prediction_model.predict_proba(normalized_points)[0]
            if np.max(class_probabilities) < 0.4:
                return 10 #significa que no hay valor
            argmax_class = np.argmax(class_probabilities)
            self.frames_prediction.appendleft(argmax_class)
            mode = Counter(self.frames_prediction).most_common(1)[0][0]
        return mode

    def get_controids(self,hand_rows):
        '''
        Calculate the centroids of the provided hand landmarks.
        Parameters
        ----------
        hand_rows : numpy.ndarray
            Array containing hand landmarks.

        Returns
        ----------
        numpy.ndarray
            Array representing the centroids of the provided hand landmarks.
        '''
        return np.mean(hand_rows, axis=0)

    def verify_hand_rows(self, right_hand_rows,left_hand_rows):
        '''
        This function validates if the landmarks can be used, it means that almost there are values for one hand
        Parameters
        ----------
        hand_rows : numpy.ndarray
            Array containing hand landmarks.

        Returns
        ----------
        numpy.ndarray
            Array representing the normalized hand landmarks.
        '''
        if (right_hand_rows is not None and left_hand_rows is not None) and (np.all(right_hand_rows != 0) or np.all(left_hand_rows != 0)): 
            return True
        else:
            return False

    def get_normalized_rows(self,hand_rows):
        '''
        Normalize the provided hand landmarks with respect to their centroids.
        Parameters
        ----------
        hand_rows : numpy.ndarray
            Array containing hand landmarks.

        Returns
        ----------
        numpy.ndarray
            Array representing the normalized hand landmarks.
        '''
        #vericar primero que hand row no sea vacio
        centroids = self.get_controids(hand_rows)
        # normalize using the calculed centroids
        normalized_points = np.array(hand_rows) - centroids
        return normalized_points