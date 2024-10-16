import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QMainWindow, QGridLayout, QComboBox

from video_thread import VideoThread
import sqlite3
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def get_app_path():
    if getattr(sys, 'frozen', False):
        app_path = os. path.dirname(sys.executable)
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))
    return app_path

app_path = get_app_path()

db_path = os.path.join(app_path, 'DataBase')
db_path = os.path.join(db_path, 'HandHygiene_database.db')
model_path = os.path.join(app_path, 'Models')
model_path = os.path.join(model_path, 'lr_10102024.pkl')

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Hygiene")
        self.showFullScreen()
        
        self.display_width = 1280
        self.display_height = 720
        self.current_frame = None
        self.video_source = None  # Initialize video source attribute
        self.video_thread = None
        # Step 1: Detect available cameras before showing the UI
        available_cameras = self.get_available_cameras()
        
        self.real_steps = [1, 2.1, 2.2, 3, 4.1, 4.2, 5.1, 5.2, 6.1, 6.2, 7]
        self.initUI(available_cameras)

        self.video_thread = VideoThread(model_path)
        self.video_thread.change_pixmap_signal.connect(self.update_image)

    def initUI(self, available_cameras):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(self.display_width, self.display_height)

        self.radio_video = QRadioButton("Video")
        self.radio_real_time = QRadioButton("Real Time")
        self.radio_video.setChecked(True)

        self.choose_button = QPushButton("Choose Video")
        self.start_stop_button = QPushButton("Start")
        self.restart_button = QPushButton("Restart")

        # Step 2: Initialize the camera selector with the available cameras
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(available_cameras)

        self.table_widget = QTableWidget(0, 2)
        self.table_widget.setHorizontalHeaderLabels(["Statistic", "Value"])

        self.traffic_light_label = QLabel(self)
        pixmap = QPixmap("/Users/juannquinones/Library/CloudStorage/OneDrive-ESCUELACOLOMBIANADEINGENIERIAJULIOGARAVITO/Nico/Manos/HigieneManos/app/semaforo.png")  # Cambia la ruta a tu imagen
        pixmap = pixmap.scaled(int(pixmap.width() * 0.25), int(pixmap.height() * 0.25), Qt.KeepAspectRatio)
        self.traffic_light_label.setPixmap(pixmap)
        circle_layout = QVBoxLayout()
        circle_layout.setSpacing(11)
        for _ in range(3):
            circle = QLabel(self)
            circle.setStyleSheet("background-color: red; border-radius: 20px;")  # 12px para hacer el círculo
            circle.setFixedSize(50, 55)  # Asigna el tamaño al QLabel
            circle_layout.addWidget(circle, alignment=Qt.AlignCenter)
            circle_layout.setContentsMargins(0, 35, 80, 0)

        self.apply_styles()

        title_label = QLabel("Hand Hygiene Video Analysis", self)
        title_label.setFont(QFont("Arial", 26, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)

        layout = QGridLayout()
        layout.addWidget(title_label, 0, 0, 1, 3, Qt.AlignTop)
        layout.addWidget(self.image_label, 1, 0, 1, 3, Qt.AlignTop)
        layout.addWidget(self.traffic_light_label, 1, 0, 1, 3, alignment=Qt.AlignTop | Qt.AlignRight)
        layout.addLayout(circle_layout, 1, 0, 1, 3, alignment=Qt.AlignTop | Qt.AlignRight)
        

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.radio_video)
        button_layout.addWidget(self.radio_real_time)
        button_layout.addWidget(self.camera_selector)  # Add camera selector to the layout
        button_layout.addWidget(self.choose_button)
        button_layout.addWidget(self.start_stop_button)
        button_layout.addWidget(self.restart_button)


        layout.addLayout(button_layout, 1, 0, 1, 3, Qt.AlignBottom)
        layout.addWidget(self.table_widget, 3, 0, 1, 3, Qt.AlignTop)

        central_widget.setLayout(layout)

        self.radio_real_time.toggled.connect(self.update_mode)
        self.radio_video.toggled.connect(self.update_mode)
        self.choose_button.clicked.connect(self.choose_video)
        self.start_stop_button.clicked.connect(self.start_stop)
        self.restart_button.clicked.connect(self.restart)

        self.update_mode()  # Ensure buttons are in correct state at start

    def apply_styles(self):
        button_style = """
            QPushButton {
                background-color: #e57373; 
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #f8bbd0;
                color: #FFFFFF;
            }
        """
        self.choose_button.setStyleSheet(button_style)
        self.start_stop_button.setStyleSheet(button_style)
        self.restart_button.setStyleSheet(button_style)

        radio_style = """
            QRadioButton {
                font-size: 18px;
                font-weight: bold;
                color: #b71c1c;
            }
        """
        self.radio_video.setStyleSheet(radio_style)
        self.radio_real_time.setStyleSheet(radio_style)

        self.camera_selector.setStyleSheet("""
            QComboBox {
                font-size: 18px;
                font-weight: bold;
                color: #b71c1c;
            }
        """)

        table_style = """
            QTableWidget {
                background-color: #ffebee;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
            }
            QHeaderView::section {
                background-color: #e57373;
                color: white;
                font-weight: bold;
                border: none;
                padding: 5px;
                font-size: 18px;
            }
            QTableWidget QTableCornerButton::section {
                background-color: #e57373;
            }
        """
        self.table_widget.setStyleSheet(table_style)
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)

    def get_available_cameras(self):
        """Detect available cameras."""
        available_cameras = []
        for i in range(10):  # Attempt to detect up to 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        return available_cameras if available_cameras else ["No Cameras Available"]

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        self.current_frame = cv_img
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(int(self.display_width * 0.8), int(self.display_height*0.8), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_mode(self):
        if self.radio_real_time.isChecked():
            if self.video_thread is not None:
                self.restart_on_change()
            self.choose_button.setEnabled(False)
            self.start_stop_button.setEnabled(True)
            # Set the selected camera as the source
            selected_camera_index = self.camera_selector.currentIndex()
            self.video_source = selected_camera_index
            self.video_thread.set_source(selected_camera_index)
        else:
            if self.video_thread is not None:
                self.restart_on_change()
            self.choose_button.setEnabled(True)
            self.start_stop_button.setEnabled(self.video_source is not None)

    def choose_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_source = file_name
            self.video_thread.set_source(file_name)
            self.start_stop_button.setEnabled(True)
            #self.radio_video.setChecked(True)

    def start_stop(self):
        if self.start_stop_button.text() == "Start":
            print('Empieza a correr el video en la fuente: ', self.video_source, 'video thread', self.video_thread._source)
            self.start_stop_button.setText("Stop")
            self.restart_button.setEnabled(False)
            self.video_thread.start()
        else:
            self.video_thread.stop()
            print('Video Detenido')
            self.start_stop_button.setText("Start")
            self.restart_button.setEnabled(True)
            #print('vector de tiempos:', self.video_thread.get_steps_times())
            record_id = self.get_lastid(self.video_thread.get_steps_times())
            #vector_save = {"Step " + str(i) +" Duration":"{:.2f}".format(v) for i,v in enumerate(self.video_thread.get_steps_times())}
            vector_save = {"Step " + str(step) + " Duration": "{:.2f}".format(v) for step, v in zip(self.real_steps, self.video_thread.get_steps_times())}
            vector_save['Total Time']=sum(self.video_thread.get_steps_times())
            
            if self.current_frame is not None:
                #HAY QUE TENER EN ESTE ScRIPTS DATOS GLOBALES MIENTRAS SE CAPTURA LA IMAGEN, Y AQUI ES CUANDO SE MUESTRAN EN LA TABLA
                self.update_statistics(vector_save,record_id)

    def restart(self):
        self.video_thread.stop()
        self.video_thread = VideoThread(model_path)  # Recreate video thread to ensure fresh start
        self.video_thread.set_source(self.video_source) #Lo crea con la configuracion anterior
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.table_widget.setRowCount(0)
        self.image_label.clear()
    
    def restart_on_change(self):
        self.video_thread.stop()
        self.video_thread = VideoThread(model_path)  # Recreate video thread to ensure fresh start
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.table_widget.setRowCount(0)
        self.image_label.clear()
        self.start_stop_button.setText("Start")

    def update_statistics(self, stats, record_id):
        self.table_widget.setRowCount(0)

        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(f"ID"))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(f"{record_id}"))

        for key, value in stats.items():
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            self.table_widget.setItem(row_position, 0, QTableWidgetItem(key))
            self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(value)))
            
    def get_lastid(self, values):
    # Verificar que el vector tenga exactamente 7 elementos
        if len(values) != 11:
            raise ValueError("The input vector must contain exactly 11 float numbers.")
        
        # Obtener la fecha y hora actual
        date_time = datetime.now()#.strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL mode
        cursor = conn.cursor()
        # Insertar el nuevo registro en la tabla
        cursor.execute('''
            INSERT INTO my_table (date_time, Step_1, "Step_2.1","Step_2.2", Step_3, "Step_4.1", "Step_4.2", "Step_5.1", "Step_5.2", "Step_6.1", "Step_6.2", Step_7)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date_time,*values))
        
        # Confirmar la transacción
        id = cursor.lastrowid
        conn.commit()
        conn.close()
        # Obtener el ID de la última fila insertada
        return id
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
