import cv2
import numpy as np 

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


(h, w) = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)

net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.4:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        print(x1, y1)
        print(x2, y2)

        #cv2.rectangle(img, (x1,y1), (x2, y2), (0, 255, 0), 2)
            

#cv2.imshow('photu', img)
#cv2.waitKey(0)

#cv2.destroyAllWindows()