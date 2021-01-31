import numpy as np
import cv2
from keras.preprocessing import image

#-----------------------------
#DNN Face Detection initialization

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#-----------------------------
def findFaces(img):
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    faces = []
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * \
                np.array([img_width, img_height, img_width, img_height])
            # (x1, y1, x2, y2) = box.astype("int")
            faces.append(box.astype("int"))
    return faces


#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

cap = cv2.VideoCapture(1)

while(True):
    ret, img = cap.read()
    (img_height, img_width) = img.shape[:2]
    #img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

    faces = findFaces(img)

    #print(faces) #locations of detected faces

    for (x1,y1,x2,y2) in faces:

        x = x1
        y = y1
        w = x2-x1
        h = y2-y1

        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
        
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
        
        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        
        emotion = emotions[max_index]
        
        #write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        #process on detected face end
        #-------------------------

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()