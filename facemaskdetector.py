import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
import dlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np



cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
result = cv2.VideoWriter('D:/arpit/maskdetection/mask.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
detector = dlib.get_frontal_face_detector()
maskpredictor =  load_model("D:/arpit/maskdetection/facemaskdetectorr.model")
prototxtPath = "D:/arpit/maskdetection/face_detector/deploy.prototxt"
weightsPath = "D:/arpit/maskdetection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
facepredict = cv2.dnn.readNet(prototxtPath, weightsPath)

def predict(frame,facepredict,maskpredictor):
    p = (0,0)
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0, 177.0, 123.0))

    facepredict.setInput(blob)
    detect = facepredict.forward()
    faces = []
    locs = []
    for i in range(0,detect.shape[2]):
        confidence = detect[0,0,i,2]
        if confidence > 0.5:
            box = detect[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype("int")
            (x1,y1) = (max(0,x1),max(0,y1))
            (x2,y2) = (min(w - 1, x2), min(h - 1, y2))

            face = frame[y1:y2,x1:x2]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((x1,y1,x2,y2))
    if len(faces)>0:
        faces = np.array(faces, dtype="float32")
        p = maskpredictor.predict(faces,batch_size = 32)

    return (locs,p)


while True:
    _,frame = cap.read()
    
    (l,p) = predict(frame,facepredict,maskpredictor)
    
    for (box,preds) in zip(l,p):
        x1,y1,x2,y2 = box
        (mask,without_mask) = preds

        label = "MASK" if mask > without_mask else "No MASK"
        color = (0,255,0) if label == "MASK" else (0,0,255)

        label = "{} : {:.2f}".format(label,max(mask,without_mask)*100)

        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,2,color,2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        print(l[0],p[0])
   
    result.write(frame) 

    cv2.imshow("img",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
