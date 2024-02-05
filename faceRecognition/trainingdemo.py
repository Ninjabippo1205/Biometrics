import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
path="faceRecognition\\datasets"
names={}

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    
    for imagePaths in imagePath:
        #faceImage = Image.open(imagePaths).convert('L')
        faceImage = cv2.imread(imagePaths, cv2.IMREAD_GRAYSCALE)
        faceNP = np.array(faceImage)
        Id= (os.path.split(imagePaths)[-1].split(".")[1])
        Id=int(Id)
        name = (os.path.split(imagePaths)[-1].split(".")[3])
        faces.append(faceNP)
        ids.append(Id)
        names[Id]=name
        cv2.imshow("Training",faceNP)
        cv2.waitKey(1)
    return ids, faces

IDs, facedata = getImageID(path)
recognizer.train(facedata, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed............")