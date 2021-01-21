import imutils
import numpy as np
import cv2 as cv
import math
from scipy.spatial import distance as dist

labels = open('../Yolov3/coco.names').read().strip().split('\n')

weights_path = "../Yolov3/yolov3.weights"
config_path = "../Yolov3/yolov3.cfg"
prob_min = 0.3
thresh = 0.3

net = cv.dnn.readNetFromDarknet(config_path, weights_path)
layers_names = net.getLayerNames()
layers_names_output = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vc = cv.VideoCapture('input/pedestrian.mp4')

frame_width = int(vc.get(3))
frame_height = int(vc.get(4))

size = (frame_width, frame_height)
fourcc = cv.VideoWriter_fourcc(*'MJPG')
resV = cv.VideoWriter('output/ped.avi', fourcc, 25, size)


def detect(frame, net, layers_names_output):
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_net = net.forward(layers_names_output)
    bounding_boxes = []
    confidences = []
    # class_numbers = []
    centroids = []
    h, w = frame.shape[:2]
    # print(h, w)

    for result in output_net:
        # print(result.shape)
        for detection in result:
            # print(x.shape)
            scores = detection[5:]
            class_id = np.argmax(scores)
            current_confidence = scores[class_id]

            if current_confidence > prob_min and class_id == 0:
                current_box = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, width, height = current_box.astype('int')
                x_min = (x_center - width / 2)
                y_min = (y_center - height / 2)
                bounding_boxes.append([int(x_min), int(y_min), int(width), int(height)])
                confidences.append(float(current_confidence))
                # class_numbers.append(class_id)
                centroids.append((x_center, y_center))

    idxs = cv.dnn.NMSBoxes(bounding_boxes, confidences, prob_min, thresh)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (bounding_boxes[i][0], bounding_boxes[i][1])
            (w, h) = (bounding_boxes[i][2], bounding_boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results


while True:
    ret, frame = vc.read()
    if not ret:
        break
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # frame = imutils.resize(frame, width=700)
    result = detect(frame, net, layers_names_output)
    nsd = set()
    if len(result) >= 2:
        centroids = np.array([r[2] for r in result])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < 100:
                    nsd.add(i)
                    nsd.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(result):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        if i in nsd:
            color = (0, 0, 255)
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv.circle(frame, (cX, cY), 5, color, 1)

    resV.write(frame)
    cv.imshow("Social Distancing Detector", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
resV.release()
cv.destroyAllWindows()
