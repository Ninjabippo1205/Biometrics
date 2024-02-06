import argparse, random

import cv2, os, json
import numpy as np

def processDataset(path, gallery_percentage, unkown_percentage, nounknown):
    if not os.path.exists(path):
        print("Dataset could not be loaded, please check folder location")
        exit(-1)

    if os.path.exists('train.json'):
        file = open('train.json', 'r')
        dataset = json.load(file)
        file.close()
        return dataset

    gallery = {}
    probe = {}
    unknown = {}

    subjects = os.listdir(path)
    if not nounknown:
        while len(unknown) < int(len(subjects)*unkown_percentage):
            item = random.randint(0, len(subjects)-1)
            unknown[subjects[item]] = os.listdir(f'{path}/{subjects[item]}')
            subjects.remove(subjects[item])

    for subject in subjects:
        files = os.listdir(f'{path}/{subject}')
        gallery[subject] = files[:int(len(files)*gallery_percentage)]
        probe[subject] = files[int(len(files)*gallery_percentage):]
    
    file = open('gallery.json', 'w')
    json.dump(gallery, file)
    file.close()

    file = open('probe.json', 'w')
    json.dump(probe, file)
    file.close()

    file = open('unknown.json', 'w')
    json.dump(unknown, file)
    file.close()

    return gallery

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
    
    parser.add_argument('-p', '--path', dest="dataset_path", type=str, default="CASIA-FaceV5", help="Path to the folder where the face images are saved")
    parser.add_argument('-gp', '--gallery-percentage', dest="gallery_percentage", default="0.8", type=float, help="Percentage of images for chosen subjects that will be added to the gallery")
    parser.add_argument('-up', '--unkown-percentage', dest="unknown_percentage", default="0.2", type=float, help="Percentage of unkown subjects that are going to be tested" )
    parser.add_argument('-nu', '--no-unknown', dest="no_unknown", action='store_true', default=False, help="Use if you'd like to add all subjects in the gallery. If this flag is set, \'unknown percentage\' will be ignored")
    arguments = parser.parse_args()

    if not os.path.exists(arguments.dataset_path):
        print("Path folder not found.")
        exit(-1)

    if arguments.gallery_percentage < 0 or arguments.gallery_percentage >= 1:
        print("Argument size not valid. Please provide a decimal number between 0 and 0.9")
        exit(-1)
    

    if arguments.unknown_percentage < 0 or arguments.unknown_percentage >= 1:
        print("Argument size not valid. Please provide a decimal number between 0 and 0.9")
        exit(-1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    gallery = processDataset(arguments.dataset_path, arguments.gallery_percentage, arguments.unknown_percentage, arguments.no_unknown)
    IDs, faceData = getImages(gallery, arguments.dataset_path)

    recognizer.train(faceData, np.array(IDs))
    print("-------------- Gallery training complete ------------")
    recognizer.write('Trainer.yml')
    print("----------------- Saved trained model ---------------")
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()