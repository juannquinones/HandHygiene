import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QMainWindow, QGridLayout, QComboBox

from video_thread import VideoThread

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

        self.initUI(available_cameras)

        self.video_thread = VideoThread()
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

        self.apply_styles()

        title_label = QLabel("Hand Hygiene Video Analysis", self)
        title_label.setFont(QFont("Arial", 26, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)

        layout = QGridLayout()
        layout.addWidget(title_label, 0, 0, 1, 3, Qt.AlignCenter)
        layout.addWidget(self.image_label, 1, 0, 1, 3, Qt.AlignCenter)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.radio_video)
        button_layout.addWidget(self.radio_real_time)
        button_layout.addWidget(self.camera_selector)  # Add camera selector to the layout
        button_layout.addWidget(self.choose_button)
        button_layout.addWidget(self.start_stop_button)
        button_layout.addWidget(self.restart_button)

        layout.addLayout(button_layout, 2, 0, 1, 3)
        layout.addWidget(self.table_widget, 3, 0, 1, 3)

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
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_mode(self):
        if self.radio_real_time.isChecked():
            if self.video_thread is not None:
                self.restart_on_change()
            self.choose_button.setEnabled(False)
            self.start_stop_button.setEnabled(True)
            # Set the selected camera as the source
            selected_camera_index = self.camera_selector.currentIndex()
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
            self.radio_video.setChecked(True)

    def start_stop(self):
        if self.start_stop_button.text() == "Start":
            self.start_stop_button.setText("Stop")
            self.restart_button.setEnabled(False)
            self.video_thread.start()
        else:
            self.video_thread.stop()
            self.start_stop_button.setText("Start")
            self.restart_button.setEnabled(True)
            if self.current_frame is not None:
                #HAY QUE TENER EN ESTE ScRIPTS DATOS GLOBALES MIENTRAS SE CAPTURA LA IMAGEN, Y AQUI ES CUANDO SE MUESTRAN EN LA TABLA
                self.update_statistics({"Total Time": 41.5, 
                                        "Step 1 Duration": 7.1,
                                        "Step 2 Duration": 7,
                                        "Step 3 Duration": 6.5,
                                        "Step 4 Duration": 6.9,
                                        "Step 5 Duration": 6.9,
                                        "Step 6 Duration": 7.1})

    def restart(self):
        self.video_thread.stop()
        self.video_thread = VideoThread()  # Recreate video thread to ensure fresh start
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.table_widget.setRowCount(0)
        self.image_label.clear()
        self.update_mode()
        self.start_stop_button.setText("Start")
    
    def restart_on_change(self):
        self.video_thread.stop()
        self.video_thread = VideoThread()  # Recreate video thread to ensure fresh start
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.table_widget.setRowCount(0)
        self.image_label.clear()
        self.start_stop_button.setText("Start")

    def update_statistics(self, stats):
        self.table_widget.setRowCount(0)
        for key, value in stats.items():
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            self.table_widget.setItem(row_position, 0, QTableWidgetItem(key))
            self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(value)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
