import cv2
import numpy as np
import os, argparse, json

def getImageID(path, names):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
 
    for imagePaths in imagePath:
        faceImage = np.array(cv2.imread(imagePaths, cv2.IMREAD_GRAYSCALE))
        id = int((os.path.split(imagePaths)[-1].split(".")[1]))
        name = (os.path.split(imagePaths)[-1].split(".")[3])
        faces.append(faceImage)
        ids.append(id)
        names[id] = name
        cv2.imshow("Training", faceImage)
        cv2.waitKey(1)
    return ids, faces


def main():
    parser = argparse.ArgumentParser(
		prog="Model training",
		description="This program handles training for real-time face recognition base on already acquired data",
		epilog="Given the amount of subjects, this program may take a while"
	)
    
    parser.add_argument('-p', '--path', dest="dataset_path", type=str, default="datasets", help="Path to the images acquired with \"datacollect.py\"")

    arguments = parser.parse_args()
    
    # Checking path validity
    if not os.path.exists(arguments.dataset_path):
        print("The dataset path is not found.")
        if arguments.dataset_path == "datasets":
            print("Have you collected any data with \"datacollect.py\"?")
        else:
            print("The path you provided is invalid, please check the arguments given in input")
        exit(-1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    names = {}
    IDs, facedata = getImageID(arguments.dataset_path, names)

    print("---------------- Training, please wait -------------------")
    recognizer.train(facedata, np.array(IDs))

    print("--------------- Model successfully trained ---------------")
    recognizer.write("Trainer.yml")

    f = open('names.json', 'w')
    json.dump(names, f)
    f.close()

    print("------------------- Training Completed -------------------")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()