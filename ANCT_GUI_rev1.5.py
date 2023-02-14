import queue
import sys
import threading

import pyautogui
from PyQt5.QtCore import QThread, QCoreApplication
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QLabel, QLineEdit, QPushButton, QVBoxLayout,
                             QWidget)

import Test_ANCT


class CaptureThread(QThread):
    def __init__(self, parent, interval, directory_path, image_limit, cfg_path, weights_path, data_path, num_threads, confidence):
        QThread.__init__(self, parent)
        super().__init__(parent)
        self.interval = interval
        self.directory_path = directory_path
        self.image_limit = image_limit
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.data_path = data_path
        self.num_threads = num_threads
        self.confidence = confidence


app = QApplication(sys.argv)


class GUI:
    def __init__(self, q):
        self.variable1 = 0
        self.variable2 = 0
        self.q = q
        self.running = False

    def update_variables(self, variable1, variable2):
        self.variable1 = variable1
        self.variable2 = variable2
        self.q.put((variable1, variable2))

    def run(self):
        self.running = True
        while self.running:
            try:
                variable1, variable2 = self.q.get(block=False)
                # use the updated variables here

            except queue.Empty:
                self.running = False
                pass


def run_main_loop():
    # Code that needs to run in a separate thread
    q = queue.Queue()
    gui = GUI(q)
    gui.run()


thread = threading.Thread(target=run_main_loop)
thread.start()


class MyApp(QWidget):
    def __init__(self, interval, directory_path, image_limit, cfg_path, weights_path, data_path, num_threads, capture_method, confidence, parent=None):
        super().__init__(parent)
        self.interval = interval
        self.directory_path = directory_path
        self.image_limit = image_limit
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.data_path = data_path
        self.num_threads = num_threads
        self.capture_method = capture_method
        self.confidence = confidence

        self.capture_thread = Test_ANCT.ScreenCapture(self.capture_method, self.interval, self.directory_path, self.image_limit,
                                                      self.cfg_path, self.weights_path, self.data_path, self.num_threads,  self.confidence)
        self.q = queue.Queue()
        self.gui = GUI(self.q)
        # Create a vertical layout to organize the widgets
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Oz's Advanced Network Collection Tool")
        self.setGeometry(300, 300, 300, 200)

        # Performance selection
        performance_label = QLabel("Performance:", self)
        self.performance_combo_box = QComboBox(self)
        self.performance_combo_box.addItems(["Low", "Medium", "High", "Ultra"])
        self.performance_combo_box.currentIndexChanged.connect(
            self.update_performance)
        layout.addWidget(performance_label)
        layout.addWidget(self.performance_combo_box)

        # Other widgets

        # Capture method selection
        capture_method_label = QLabel("Capture Method:", self)
        self.capture_method_combo_box = QComboBox(self)
        self.capture_method_combo_box.addItems(
            ["Detection", "Screenshot", "Video"])
        self.capture_method_combo_box.currentIndexChanged.connect(
            self.update_capture_method)
        layout.addWidget(capture_method_label)
        layout.addWidget(self.capture_method_combo_box)

        # Selection of the resolution / image size
        resolution_label = QLabel("Resolution:", self)
        self.resolution = str(
            pyautogui.size()[0]) + "x" + str(pyautogui.size()[1])
        self.resolution_combo_box = QComboBox(self)
        self.resolution_combo_box.addItems(["1920x1080", "Random"])
        self.resolution_combo_box.currentIndexChanged.connect(
            self.update_resolution)
        layout.addWidget(resolution_label)
        layout.addWidget(self.resolution_combo_box)

        # Random image size parameters
        random_size_label = QLabel("Random Image Size:", self)
        self.random_size_line_edit = QLineEdit("250, 1920, 1080", self)
        self.random_size_line_edit.setEnabled(False)
        self.random_size_line_edit.textChanged.connect(self.update_random_size)
        layout.addWidget(random_size_label)
        layout.addWidget(self.random_size_line_edit)

        # Image limit input box
        image_limit_label = QLabel("Image Limit:", self)
        self.image_limit_line_edit = QLineEdit("1000", self)
        self.image_limit_line_edit.textChanged.connect(self.update_image_limit)
        layout.addWidget(image_limit_label)
        layout.addWidget(self.image_limit_line_edit)

        # Image limit indefinite check box
        self.image_limit_indefinite_checkbox = QCheckBox("Indefinite", self)

        # what i need to pass to next function
        self.image_limit_indefinite_checkbox.stateChanged.connect(
            self.update_image_limit_indefinite)
        layout.addWidget(self.image_limit_indefinite_checkbox)

        # Weights selection
        self.weights_button = QPushButton("Select Weights", self)
        self.weights_button.clicked.connect(self.load_weights)
        layout.addWidget(self.weights_button)

        # CFG selection
        self.cfg_button = QPushButton("Select CFG", self)
        self.cfg_button.clicked.connect(self.load_cfg)
        layout.addWidget(self.cfg_button)

        # Data selection
        self.data_button = QPushButton("Select Data", self)
        self.data_button.clicked.connect(self.load_data)
        layout.addWidget(self.data_button)

        # Sorted image directory splitting location
        self.directory_button = QPushButton("Select Directory", self)
        self.directory_button.move(190, 1)
        self.directory_button.clicked.connect(self.select_directory)
        layout.addWidget(self.data_button)

        # Start/Stop the script
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # Add an interval input line edit to the layout
        interval_label = QLabel("Interval (ms):", self)
        self.interval_line_edit = QLineEdit("1000", self)
        self.interval_line_edit.textChanged.connect(self.update_interval)
        layout.addWidget(interval_label)
        layout.addWidget(self.interval_line_edit)

        # Confidence level selection
        confidence_label = QLabel("Confidence Level:", self)
        self.confidence_line_edit = QLineEdit("0.5", self)
        self.confidence_line_edit.textChanged.connect(self.update_confidence)
        layout.addWidget(confidence_label)
        layout.addWidget(self.confidence_line_edit)

        self.show()

    def update_capture_method(self, index):
        print("Capture method changed to: " + str(index) + "' hh")
        self.capture_method = int(index)
        self.capture_thread.capture_method = int(index)
        self.gui.update_variables(self.capture_method, self.resolution)

    def update_performance(self):
        performance = self.performance_combo_box.currentText()
        if performance == "Low":
            self.capture_thread.num_threads = 1
        elif performance == "Medium":
            self.capture_thread.num_threads = 2
        elif performance == "High":
            self.capture_thread.num_threads = 4
        else:
            self.capture_thread.num_threads = 8

    def update_image_limit_indefinite(self):
        if self.image_limit_indefinite_checkbox.isChecked():
            self.image_limit_line_edit.setEnabled(False)
            self.image_limit_line_edit.setText("-1")
        else:
            self.image_limit_line_edit.setText("0")
            self.image_limit_line_edit.setEnabled(True)

    def capture_method(self):
        capture_method = self.capture_method_combo_box.currentText()
        if self.capture_method == "Detection":
            self.capture_thread.detection_capture()
        elif self.capture_method == "Video":
            self.capture_thread.video_capture()
        elif self.capture_method == "Screenshot":
            self.capture_thread.screenshot_capture()

    def update_interval(self):
        interval = (self.interval_line_edit.text())
        self.capture_thread.interval = int(interval) if (
            interval and interval.isdigit()) else 0

    def update_confidence(self):
        confidence = self.confidence_line_edit.text()
        self.capture_thread.confidence = float(confidence) if (
            confidence and confidence.isdigit()) else 0

    def update_resolution(self):
        resolution = self.resolution_combo_box.currentText()
        if resolution == "1920x1080":
            self.capture_thread.size = (1920, 1080)
            self.random_size_line_edit.setEnabled(False)
        else:
            self.random_size_line_edit.setEnabled(True)

    def update_random_size(self):
        random_size = self.random_size_line_edit.text()
        try:
            self.capture_thread.random_size = tuple(
                map(int, random_size.split(",")))
        except ValueError:
            self.random_size_line_edit.setText("250, 1920, 1080")
            self.capture_thread.random_size = (250, 1920, 1080)

    def update_image_limit(self):
        image_limit = self.image_limit_line_edit.text()
        if self.image_limit_indefinite_checkbox.isChecked():
            self.capture_thread.image_limit = (-1)
        else:
            self.capture_thread.image_limit = int(
                image_limit) if (image_limit and image_limit.isdigit()) else 0

    def load_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Weights File", "", "Weights Files (*.weights);;All Files ()", options=options)
        if file_name:
            self.capture_thread.weights_path = file_name

    def load_cfg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select CFG File", "", "CFG Files (*.cfg);;All Files ()", options=options)
        if file_name:
            self.capture_thread.cfg_path = file_name

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "Data Files (*.txt);;All Files ()", options=options)
        if file_name:
            self.capture_thread.data_path = file_name

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", "", QFileDialog.ShowDirsOnly)
        if directory:
            self.capture_thread.directory_path = directory

    def start(self):
        self.capture_thread.stopped = False

        self.capture_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop(self):
        self.capture_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == '__main__':

    my_app = MyApp(1, "target", 1000,
                   "cfg/yolov4.cfg", "weights/yolov4.weights", "data/classes.names", 1, "Screenshot", 0.6)
    app.exec_()
    sys.exit(app.exec_())
