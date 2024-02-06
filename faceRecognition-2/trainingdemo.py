import cv2, os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
path="faceRecognition\\datasets"
names={}

def createDataset():
    return []

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

def main():
    dataset = createDataset()
    IDs, faceData = getImages(dataset)

    recognizer.train(faceData, np.array(IDs))
    recognizer.write('Trainer.yml')
    cv2.destroyAllWindows()
    print("-------------- Training Complete ------------")
    return

if __name__ == "__main__":
    main()