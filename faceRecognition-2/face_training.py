import cv2, os, json
import numpy as np

def createDataset(path):
    dataset = {}

    if not os.path.exists(path):
        print("Dataset could not be loaded, please check folder location")
        exit(-1)
    
    for subject in os.listdir(path):
        dataset[subject] = os.listdir(f'{path}/{subject}')
    
    file = open('dataset.json', 'w')
    json.dump(dataset, file)
    file.close()

    return dataset

def getImages(dataset, path):
    ids = []
    faces = []

    for subject in dataset:
        for image in dataset[subject]:
            faceImage = np.array(cv2.imread(f'{path}/{subject}/{image}'))
            faces.append(faceImage)
            ids.append(subject)
            cv2.imshow("Training", faceImage)
            cv2.waitKey(1)
    return ids, faces

def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path="faceRecognition/"

    dataset = createDataset(path)
    IDs, faceData = getImages(dataset)

    recognizer.train(faceData, np.array(IDs))
    recognizer.write('Trainer.yml')
    cv2.destroyAllWindows()
    print("-------------- Training Complete ------------")
    return

if __name__ == "__main__":
    main()