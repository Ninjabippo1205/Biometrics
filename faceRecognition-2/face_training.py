import cv2, os, json
import numpy as np
import argparse

def createDataset(path, trainsize):
    if not os.path.exists(path):
        print("Dataset could not be loaded, please check folder location")
        exit(-1)

    if os.path.exists('train.json'):
        file = open('train.json', 'r')
        dataset = json.load(file)
        file.close()
        return dataset

    train = {}
    test = {}

    for subject in os.listdir(path):
        files = os.listdir(f'{path}/{subject}')
        train[subject] = files[:int(len(files)*trainsize)]
        test[subject] = files[int(len(files)*trainsize):]
    
    file = open('train.json', 'w')
    json.dump(train, file)
    file.close()

    file = open('test.json', 'w')
    json.dump(test, file)
    file.close()

    return train, test

def getImages(trainset, path):
    ids = []
    faces = []

    for subject in trainset:
        for image in trainset[subject]:
            faceImage = np.array(cv2.imread(f'{path}/{subject}/{image}', cv2.IMREAD_GRAYSCALE))
            faces.append(faceImage)
            ids.append(int(subject))
            #cv2.imshow("Training", faceImage)
            #cv2.waitKey(1)
    return ids, faces

def main():
    parser = argparse.ArgumentParser(
		prog="FaceTraining",
		description="This program handles training a model based on a given face recognition dataset",
		epilog="The dataset must be created as such: \"dataset/xxx/xxx_y.bmp\" where \'x\' is the subject's number and \'y\' the numbered photo."
	)
    
    parser.add_argument('-p', '--path', dest="path", type=str, default="CASIA-FaceV5", help="Folder where iris images are saved")
    parser.add_argument('-ts', '--train-size', dest="tsize", default="0.8", help="Amount of data used for training")
    arguments = parser.parse_args()

    if not os.path.exists(arguments.path):
        print("Path folder not found.")
        exit(-1)
    
    try:
        size = float(arguments.tsize)
        if size < 0 or size >= 1: raise ValueError
    except ValueError:
        print("Argument size not valid. Please provide a decimal number between 0 and 0.9")
        exit(-1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    train, test = createDataset(arguments.path, size)
    IDs, faceData = getImages(train, arguments.path)

    recognizer.train(faceData, np.array(IDs))
    print("-------------- Training Complete ------------")
    recognizer.write('Trainer.yml')
    print("------------- Saved Trained Model -----------")
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()