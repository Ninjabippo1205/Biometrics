import cv2, time
import os, requests, json

def main():
    # Moving inside folder (os dependent)
    current_folder = os.path.realpath(__file__)
    if os.name == 'nt': os.chdir(current_folder[:current_folder.rfind('\\')])
    elif os.name == 'posix': os.chdir(current_folder[:current_folder.rfind('/')])
    else:
        print("This program hasn't been tested in the current enviroment. Downtime may happen.")
        _ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
        if _ != '': exit(-1)
        os.chdir(current_folder[:current_folder.rfind('/')])

    # Requirements
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        content = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        open("haarcascade_frontalface_default.xml", "wb").write(content.content)

    if not os.path.exists("Trainer.yml"):
        print("Trainer.yml not found: Have you trained the model?")
        exit(-1)
    
    # Getting names
    if not os.path.exists("names.json"):
        print("names.json not found: have you trained the model?")
        exit(-1)
    
    fp = open('name.json', 'r')
    names = json.load(fp)
    fp.close()

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")

    while True:
        ret, frame = video.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray_image, 1.3, 5)
        for (x,y,w,h) in faces:
            serial, conf = recognizer.predict(gray_image[y:y+h, x:x+w])
            if conf>60:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.rectangle(frame, (x,y-40), (x+w,y), (255,0,0), -1)
                cv2.putText(frame, names[serial], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        frame=cv2.resize(frame, (640, 480))

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if key == ord('o') and conf > 50:
            time.sleep(10)
        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()