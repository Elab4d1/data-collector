import cv2
import os

classes = open('classes.names')
content = classes.readlines()
path = os. getcwd()
interval = 5
i, count = 0, 0
directory = 'videos'

with open('classes.names', 'r') as f:
    classes = f.read().splitlines()
net = cv2.dnn
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        cap = cv2.VideoCapture(f)
        success, img = cap.read()
        while success:
            cap.set(cv2.CAP_PROP_POS_MSEC, (count*interval*1000))
            classIds, scores, boxes = model.detect(
                img, confThreshold=0.6, nmsThreshold=0.4)
            uniclass = set(classIds)
            for cls in uniclass:
                if not os.path.exists(content[cls][:-1]):
                    os.mkdir(content[cls][:-1])
            for (classId, score, box) in zip(classIds, scores, boxes):
                i = i+1

                nimg = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                text = '%s: %.2f' % (classes[classId], score)
                cv2.imwrite(os.path.join(
                    path+'\\'+content[classId][:-1], content[classId][:-1]+str(i)+'.jpg'), nimg)
            success, img = cap.read()
            count += 1
        cap.release()
