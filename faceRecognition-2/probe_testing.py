import cv2, time, os
import json, requests, argparse

def main():
    parser = argparse.ArgumentParser(
		prog="FaceTesting",
		description="This program handles testing of a trained model based on a given face recognition dataset",
		epilog="The dataset must be created as such: \"dataset/xxx/xxx_y.bmp\" where \'x\' is the subject's number and \'y\' the numbered photo."
	)
    
    parser.add_argument('-dp', '--dataset-path', dest="dataset_path", type=str, default="CASIA-FaceV5", help="Path to the folder where the face images are saved")
    parser.add_argument('-pp', '--probe-path', dest="probe_path", type=str, default="probe.json", help="Path to the probe json file created in training")
    parser.add_argument('-up', '--unknown-path', dest='unknown_path', type=str, default='unknown.json', help="Path to the unkown json file created in training")
    parser.add_argument('-nu', '--no-unknown', dest="no_unknown", action='store_true', default=False, help="Use if you don't have any unknown subject to test. If this flag is set, \'unknown path\' will be ignored")

    arguments = parser.parse_args()

    if not os.path.exists(arguments.dataset_path):
        print("Images folder not found, path invalid")
        exit(-1)
    
    if not os.path.exists(arguments.probe_path):
        print("Probe dataset representation could not be found, path invalid")
        exit(-1)
    
    if not os.path.exists(arguments.unknown_path) and not arguments.no_unknown:
        print("Unknown dataset representation could not be found, however \'no unknown\' has not been set.")
        _ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
        if _ != '': exit(-1)

    if not os.path.exists('Trainer.yml'):
        print("Model \'Trainer.yml\' not found in its default path. Have you trained the model?")
        exit(-1)
    
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        content = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        open("haarcascade_frontalface_default.xml", "wb").write(content.content)
    
    with open(arguments.probe_path, 'r') as f:
        probe = json.load(f)
    
    if not arguments.no_unknown:
        with open(arguments.unknown_path, 'r') as f:
            unknown = json.load(f)

    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")

if __name__ == '__main__':
    main()