import sys
import cv2
import csv
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        self.clicked.emit(event.x(), event.y())

class VideoAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_path = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.start_point = None
        self.end_point = None
        self.movement_index = 0

    def initUI(self):
        self.setWindowTitle('Laser Pointer Video Analyzer')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Video display
        self.video_label = ClickableLabel()
        self.video_label.clicked.connect(self.on_video_clicked)
        layout.addWidget(self.video_label)

        # Slider for video navigation
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_frame)
        layout.addWidget(self.slider)

        # Status label
        self.status_label = QLabel('Load a video to start')
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Video')
        self.load_button.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_button)

        layout.addLayout(button_layout)

        main_widget.setLayout(layout)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setEnabled(True)
            self.update_frame()
            self.status_label.setText('Select start point')

    def update_frame(self):
        if self.cap is not None:
            self.current_frame = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # Draw start point if it exists
                if self.start_point:
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.red, 5))
                    painter.drawPoint(self.start_point[0], self.start_point[1])
                    painter.end()
                
                self.video_label.setPixmap(pixmap)

    def on_video_clicked(self, x, y):
        if self.start_point is None:
            self.start_point = (x, y)
            self.status_label.setText('Select end point')
        else:
            self.end_point = (x, y)
            self.calculate_and_save_heading()
            self.start_point = None
            self.end_point = None
            self.status_label.setText('Select start point for next movement')

    def calculate_and_save_heading(self):
        if self.start_point and self.end_point:
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            heading = np.arctan2(dy, dx)
            
            # Save to CSV
            video_name = self.video_path.split('/')[-1]
            with open('laser_movements.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([video_name, self.movement_index, heading])

            print(f"Saved movement {self.movement_index} with heading {heading}")
            self.movement_index += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoAnalyzer()
    ex.show()
    sys.exit(app.exec_())