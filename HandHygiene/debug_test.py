
from HandHygieneMain import *

#MODEL_PATH = r'D:\\Proyectos\\Hands\\HigieneManos\\Data\\Models\\rf.pkl'
#MODEL_PATH ='/Users/juannquinones/Library/CloudStorage/OneDrive-ESCUELACOLOMBIANADEINGENIERIAJULIOGARAVITO/Nico/Manos/HigieneManos/Data/Models/modelo.pkl'
MODEL_PATH ='avianca.pkl'
with open(MODEL_PATH, 'rb') as file:
    modelo = pickle.load(file)


cap = cv2.VideoCapture(0)
# Set video properties for optimal performance
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands = 2,static_image_mode=True) # modelo

image_success = True
hand_model = HandHygineModel(mp_drawing, mp_drawing_styles, mp_hands, hands, step_prediction_model=modelo)
while cap.isOpened() and image_success:
    image_success, image = cap.read()
    success, _, right_hand_rows, left_hand_rows = hand_model.get_landmarks_structure(success=image_success, image = image, mode='capture', return_image=True)

    if success: # Solo se procesan las que tienen landmarks validos 
        if hand_model.verify_hand_rows(right_hand_rows,left_hand_rows):
            X = np.concatenate([hand_model.get_normalized_rows(right_hand_rows), hand_model.get_normalized_rows(left_hand_rows)], axis=0).reshape(42*3) 
            #print(X.shape)
            y=hand_model.predict_hygiene_step(X.reshape(1,-1))
            print(y)

        cv2.putText(image, f"Step: {y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Hand step clasiffication in Real Time', image)

        #cv2.putText(image, str(round(y[0],1)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow('Paso 1', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

