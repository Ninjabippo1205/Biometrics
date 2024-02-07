import cv2, time, trainingdemo

video=cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf>60:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(255,0,0),-1)
            cv2.putText(frame, trainingdemo.names[serial], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, "Unknown", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    frame=cv2.resize(frame, (640, 480))
    
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    
    if k==ord('o') and conf>50:
        time.sleep(10)
    if k==ord("q"):
        break

video.release()
cv2.destroyAllWindows()