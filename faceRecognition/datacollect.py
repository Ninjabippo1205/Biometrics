import cv2

import os, requests

video=cv2.VideoCapture(0)
if not os.path.exists("haarcascade_frontalface_default.xml"):
    content = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
    open("haarcascade_frontalface_default.xml", "wb").write(content.content)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
id = input("Enter Your ID: ")
name = input("Enter Your Name: ")
count=0

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite('datasets/User.'+str(id)+"."+str(count)+"."+str(name)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)

    if count>500:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done..................")