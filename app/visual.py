
import mediapipe as mp
import time
import pickle
import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HandHygiene')))
print("sys.path:", sys.path)
from HandHygieneMain import *
import sys
from PIL import Image , ImageTk
import cv2
import imutils

import numpy as np

MODEL_PATH ='/Users/juannquinones/Library/CloudStorage/OneDrive-ESCUELACOLOMBIANADEINGENIERIAJULIOGARAVITO/Nico/Manos/HigieneManos/Data/Models/rf_260624.pkl'
#MODEL_PATH ='rf_5es_98acc.pkl'
with open(MODEL_PATH, 'rb') as file:
    modelo = pickle.load(file)
step_time =[0,0,0,0,0,0,0]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands = 2,static_image_mode=True)
hand_model = HandHygineModel(mp_drawing, mp_drawing_styles, mp_hands, hands, step_prediction_model=modelo)
image_success = True

# Configuración de la interfaz gráfica
class Interfaz(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Form")
        self.resize(600, 500)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.setStyleSheet("background-color:rgb(243, 184, 184)")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalWidget = QtWidgets.QWidget(self)
        self.verticalWidget.setObjectName("verticalWidget")
        self.verticalLayoutImagen = QtWidgets.QVBoxLayout(self.verticalWidget)
        self.verticalLayoutImagen.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayoutImagen.setObjectName("verticalLayoutImagen")
        self.horizontalWidgetHeader = QtWidgets.QWidget(self.verticalWidget)
        self.horizontalWidgetHeader.setStyleSheet("background-color: rgb(206, 42, 41)")
        self.horizontalWidgetHeader.setObjectName("horizontalWidgetHeader")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidgetHeader)
        self.horizontalLayout.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.horizontalWidgetHeader)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setContextMenuPolicy(Qt.NoContextMenu)
        self.label_2.setLayoutDirection(Qt.LeftToRight)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setText("")
        image = QtGui.QPixmap("logo_shaio.png")
        self.label_2.setPixmap(image.scaled(140, 140, Qt.KeepAspectRatio))
        self.label_2.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.horizontalWidgetHeader)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setLayoutDirection(Qt.LeftToRight)
        self.label.setStyleSheet("color:white")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_3 = QtWidgets.QLabel(self.horizontalWidgetHeader)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.verticalLayoutImagen.addWidget(self.horizontalWidgetHeader)
        self.verticalLayout_2.addWidget(self.verticalWidget)
        self.image_label = QtWidgets.QLabel(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMinimumSize(QtCore.QSize(600, 300))
        self.image_label.setStyleSheet("background-color:white; border-width:3px; border-color:rgb(0, 0, 0); border-style:inset;")
        self.image_label.setText("")
        self.image_label.setObjectName("image_label")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.verticalLayout_2.addWidget(self.image_label)
        self.horizontalLayoutButtons = QtWidgets.QHBoxLayout()
        self.horizontalLayoutButtons.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayoutButtons.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayoutButtons.setSpacing(10)
        self.horizontalLayoutButtons.setObjectName("horizontalLayoutButtons")
        self.pushButton_real_time = QtWidgets.QPushButton(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_real_time.sizePolicy().hasHeightForWidth())
        self.pushButton_real_time.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_real_time.setFont(font)
        self.pushButton_real_time.setStyleSheet("background-color:rgb(206, 42, 41); color: white;")
        self.pushButton_real_time.setObjectName("pushButton_real_time")
        self.horizontalLayoutButtons.addWidget(self.pushButton_real_time)
        self.pushButton_local_video = QtWidgets.QPushButton(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_local_video.sizePolicy().hasHeightForWidth())
        self.pushButton_local_video.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_local_video.setFont(font)
        self.pushButton_local_video.setStyleSheet("background-color:rgb(206, 42, 41); color: white;")
        self.pushButton_local_video.setObjectName("pushButton_local_video")
        self.horizontalLayoutButtons.addWidget(self.pushButton_local_video)
        self.pushButton_reset = QtWidgets.QPushButton(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_reset.sizePolicy().hasHeightForWidth())
        self.pushButton_reset.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_reset.setFont(font)
        self.pushButton_reset.setStyleSheet("background-color:rgb(206, 42, 41); color: white;")
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.horizontalLayoutButtons.addWidget(self.pushButton_reset)
        self.verticalLayout_2.addLayout(self.horizontalLayoutButtons)
        self.timer_label = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.timer_label.setFont(font)
        self.timer_label.setStyleSheet("background-color:white; border-width:2px; border-color:rgb(0, 0, 0); border-style:solid; padding:5px;")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setObjectName("timer_label")
        self.verticalLayout_2.addWidget(self.timer_label)
        self.pushButton_real_time.clicked.connect(self.change_button_text_real_time)
        self.pushButton_local_video.clicked.connect(self.change_button_text_local_video)
        self.setWindowTitle("Hand Higyene Interfaz")
        self.retranslateUi()
    def change_button_text_real_time(self):
        if self.pushButton_real_time.text() == "Real-Time":
            self.pushButton_real_time.setText("Finish")
            self.pushButton_local_video.setText("Save Data")
        elif self.pushButton_real_time.text() == "Finish":
            self.pushButton_real_time.setText("Real-Time")
            self.pushButton_local_video.setText("Load Video")

    def change_button_text_local_video(self):
        if self.pushButton_local_video.text() == "Load Video":
            self.pushButton_local_video.setText("Finish")
            self.pushButton_real_time.setText("Save Data")
        elif self.pushButton_local_video.text() == "Finish":
            self.pushButton_local_video.setText("Load Video")
            self.pushButton_real_time.setText("Real-Time")


    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.pushButton_real_time.setText(_translate("Form", "Real-Time"))
        self.pushButton_local_video.setText(_translate("Form", "Load Video"))
        self.pushButton_reset.setText(_translate("Form", "Restart"))
        self.timer_label.setText(_translate("Form", "00:00:00"))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = Interfaz()
    window.show()
    sys.exit(app.exec_())
