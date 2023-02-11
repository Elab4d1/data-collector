import cv2
import os
import uuid
import numpy as np
import queue
import random
import mss
import time
import pyautogui
from PyQt5.QtCore import QThread, pyqtSignal
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import shutil
import threading

path = os.getcwd()


class CaptureThread(QThread):
    processed_images_count = 0
    update_progress = pyqtSignal(int)


class ScreenCapture:
    def __init__(self, capture_method, interval, directory_path, image_limit, cfg_path, weights_path, data_path, num_threads, confidence):
        super().__init__()
        self.lock = threading.Lock()
        self.capture_methods = {
            "Detection": self.detection_capture,
            "Video": self.video_capture,
            "Screenshot": self.screenshot_capture
        }
        self.capture_method = capture_method
        self.interval = interval
        self.directory_path = directory_path
        self.image_limit = image_limit
        self.confidence = confidence
        self.size = (1920, 1080)
        self.random_size = (250, 1920, 1080)
        self.location = (0, 0)
        self.stopped = False
        self.image_queue = queue.Queue()
        self.num_threads = num_threads
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg_path = os.path.join(current_dir, "cfg/yolov4.cfg")
        if not os.path.exists("target"):
            os.makedirs("target")
        if not os.path.exists(cfg_path):
            print(f"cfg_path does not exist: {cfg_path}")
        self.weights_path = os.path.join(
            current_dir, "weights/yolov4.weights")
        if not os.path.exists(weights_path):
            print(f"weights_path does not exist: {weights_path}")
        self.data_path = os.path.join(current_dir, "data/classes.names")
        if not os.path.exists(data_path):
            print(f"data_path does not exist: {data_path}")
        self.net = cv2.dnn.readNet(self.cfg_path, self.weights_path)
        self.classes = None
        with open(self.data_path, "r") as f:
            self.classes = f.readlines()
            self.layer_names = self.net.getLayerNames()

        self.output_layers = [self.layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]

    def run(self):
        self.screenshot_capture()
        # self.capture_methods[self.capture_method]()

    def start(self):
        thread = Thread(target=self.run)
        thread.start()

    def detection_capture(self):
        max_width, max_height = pyautogui.size()
        for i in range(self.image_limit):
            if self.stopped:
                break

            width, height = None, None
            x, y = None, None
            if self.size == "Random":
                width = random.randint(self.random_size[0], min(
                    self.random_size[1], max_width))
                height = random.randint(self.random_size[0], min(
                    self.random_size[2], max_height))
                x = random.randint(0, max_width - width)
                y = random.randint(0, max_height - height)
            else:
                width, height = self.size
                x, y = self.location

            if width < 250 or height < 250:
                width, height = 250, 250

            screen = np.array(cv2.VideoCapture(0).read()[1])
            mobileNetSSDImgSize = (416, 416)
            img = cv2.resize(screen, mobileNetSSDImgSize)
            cv2.imshow("Resized image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            blob = cv2.dnn.blobFromImage(
                img, 1.0/255.0, mobileNetSSDImgSize, swapRB=True, crop=False)
            self.net.setInput(blob)
            detections = self.net.forward()
            print(detections.shape)
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.1:
                    idx = int(detections[0, 0, i, 1])
                    x_det = int(detections[0, 0, i, 3] * img.shape[1])
                    y_det = int(detections[0, 0, i, 4] * img.shape[0])
                    w_det = int(detections[0, 0, i, 5] * img.shape[1])
                    h_det = int(detections[0, 0, i, 6] * img.shape[0])
                    if self.classes:
                        label = self.classes[idx]
                        cv2.rectangle(
                            img, (x_det, y_det), (x_det + w_det, y_det + h_det), (0, 255, 0), 2)
                        cv2.putText(img, label, (x_det, y_det - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # take screenshot of detection region
                    pyautogui.screenshot(
                        region=(x+x_det, y+y_det, w_det, h_det))

            unique_filename = str(uuid.uuid4()) + ".jpg"
            img_path = os.path.join(self.directory_path, unique_filename)
            cv2.imwrite(img_path, img)
            self.image_queue.put((img_path, 0))
            self.processed_images_count += 1
            self.update_progress.emit(self.processed_images_count)
            # added cap var so the release works
            cap = cv2.VideoCapture(0)
            if self.interval > 0:
                time.sleep(self.interval)
            cap.release()
            cv2.destroyAllWindows()

    def video_capture(self):
        # added max_width and max_height
        max_width, max_height = pyautogui.size()
        # Get list of all video files in directory
        videos = [f for f in os.listdir(
            self.directory_path) if f.endswith('.mp4')]
        for video in videos:
            cap = cv2.VideoCapture(os.path.join(self.directory_path, video))
            while True:
                if self.stopped:
                    break
                if self.processed_images_count >= self.image_limit:
                    self.stopped = True
                    # Show alert or dialog box
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                    # Run the object detector on the frame
                width, height = None, None
                x, y = None, None
                if self.size == "Random":
                    width = random.randint(self.random_size[0], min(
                        self.random_size[1], max_width))
                    height = random.randint(self.random_size[0], min(
                        self.random_size[2], max_height))
                    x = random.randint(0, max_width - width)
                    y = random.randint(0, max_height - height)
                else:
                    width, height = self.size
                    x, y = self.location

                if width < 250 or height < 250:
                    width, height = 250, 250
                frame = frame[y:y+height, x:x+width]
                blob = cv2.dnn.blobFromImage(
                    frame, 1.0/255.0, (16, 16), swapRB=True, crop=False)
                self.net.setInput(blob)
                detections = self.net.forward()
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.1:
                        idx = int(detections[0, 0, i, 1])
                        x = int(detections[0, 0, i, 3] * frame.shape[1])
                        y = int(detections[0, 0, i, 4] * frame.shape[0])
                        w = int(detections[0, 0, i, 5] * frame.shape[1])
                        h = int(detections[0, 0, i, 6] * frame.shape[0])
                        if self.classes:
                            label = self.classes[idx]
                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                unique_filename = str(uuid.uuid4()) + ".jpg"
                img_path = os.path.join(self.directory_path, unique_filename)
                img = cv2.imread(img_path)
                cv2.imwrite(img_path, img)
                self.image_queue.put((img_path, 0))
                self.processed_images_count += 1
                self.update_progress.emit(self.processed_images_count)

                if self.interval > 0:
                    time.sleep(self.interval)
                cap.release()
                cv2.destroyAllWindows()

    def screenshot_capture(self):
        model = cv2.dnn_DetectionModel(self.net)
        model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        while not self.stopped and self.processed_images_count < self.image_limit:
            self.processed_images_count += 1
            myScreenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(myScreenshot), cv2.COLOR_RGB2BGR)
            classIds, scores, boxes = model.detect(
                img, confThreshold=self.confidence, nmsThreshold=0.4)
            uniclass = set(classIds)
            for cls in uniclass:
                print(os.path.join(
                    path, self.directory_path, self.classes[cls][:-1]))
                if not os.path.exists(os.path.join(path, self.directory_path, self.classes[cls][:-1])):
                    os.mkdir(os.path.join(
                        path, self.directory_path, self.classes[cls][:-1]))
            for (classId, score, box) in zip(classIds, scores, boxes):

                img = cv2.cvtColor(np.array(myScreenshot),
                                   cv2.COLOR_RGB2BGR)
                # cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                #               color=(0, 255, 0), thickness=2)

                nimg = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                text = '%s: %.2f' % (self.classes[classId], score)
                cv2.putText(img, text, (box[0]+17, box[1]+17), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 0), thickness=2)
                cv2.imwrite(os.path.join(
                    path, self.directory_path, self.classes[classId][:-1], str(uuid.uuid4())+'.jpg'), nimg)
                time.sleep(self.interval)

    def process_sort(self, img_path, percentage):
        new_path = None
        # sort images into folders based on the confidence percentage
        if percentage < 15:
            if not os.path.exists("0-15%"):
                os.makedirs("0-15%")
            new_path = f"0-15%/{str(uuid.uuid4())}.jpg"
        elif percentage < 35:
            if not os.path.exists("15-35%"):
                os.makedirs("15-35%")
            new_path = f"15-35%/{str(uuid.uuid4())}.jpg"
        elif percentage < 60:
            if not os.path.exists("35-60%"):
                os.makedirs("35-60%")
            new_path = f"35-60%/{str(uuid.uuid4())}.jpg"
        elif percentage < 80:
            if not os.path.exists("60-80%"):
                os.makedirs("60-80%")
            new_path = f"60-80%/{str(uuid.uuid4())}.jpg"
        elif percentage <= 100:
            if not os.path.exists("80-100%"):
                os.makedirs("80-100%")
            new_path = f"80-100%/{str(uuid.uuid4())}.jpg"
        else:
            if not os.path.exists("Manual_Review"):
                os.makedirs("Manual_Review")
            new_path = f"Manual_Review/{str(uuid.uuid4())}.jpg"
        shutil.move(img_path, new_path)

    def stop(self):
        self.stopped = True
