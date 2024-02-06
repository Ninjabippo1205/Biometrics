import cv2, time, os
import json, requests, argparse
import face_training

def main():
    parser = argparse.ArgumentParser(
		prog="FaceTesting",
		description="This program handles testing of a trained model based on a given face recognition dataset",
		epilog="The dataset must be created as such: \"dataset/xxx/xxx_y.bmp\" where \'x\' is the subject's number and \'y\' the numbered photo."
	)
    
    parser.add_argument('-p', '--path', dest="path", type=str, default="CASIA-FaceV5", help="Folder where iris images are saved")
    parser.add_argument('-td', '--test-dataset', dest="testset", default="test.json", help="File that holds the test portion of the model")
    arguments = parser.parse_args()

    if not os.path.exists(arguments.path):
        print("Path folder not found.")
        exit(-1)
    
    if not os.path.exists(arguments.testset):
        print("Test dataset representation could not be found")
        exit(-1)

    if not os.path.exists('Trainer.yml'):
        print("Please proceed to train the model first")
        exit(-1)
    
    with open(arguments.testset) as f:
        testset = json.load(f) 

    if not os.path.exists("haarcascade_frontalface_default.xml"):
        content = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        open("haarcascade_frontalface_default.xml", "wb").write(content.content)

    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")

    TP = 0; TN = 0; FP = 0; FN = 0

    for subject in testset:
        for image in testset[subject]:
            faceImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_training.detectMultiScale(faceImage, 1.3, 5)

            for (x,y,w,h) in faces:
                serial, conf = recognizer.predict(faceImage[y:y+h, x:x+w])

                print(serial, conf)


#                if conf>60:
#                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#                    cv2.rectangle(frame,(x,y-40),(x+w,y),(255,0,0),-1)
#                    cv2.putText(frame, trainingdemo.names[serial], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
#                else:
#                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#                    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#                    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#                    cv2.putText(frame, "Unknown", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
if __name__ == '__main__':
    main()