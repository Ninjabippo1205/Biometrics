import cv2
import os, requests, argparse


def main():
    # Moving inside folder (os dependent)
    current_folder = os.path.realpath(__file__)
    if os.name == 'nt': os.chdir(current_folder[:current_folder.rfind('\\')])
    elif os.name == 'posix': os.chdir(current_folder[:current_folder.rfind('/')])
    else:
        print("This program hasn't been tested in the current enviroment. Downtime may occur.")
        _ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
        if _ != '': exit(-1)
        os.chdir(current_folder[:current_folder.rfind('/')])

    parser = argparse.ArgumentParser(
		prog="Data collcetion",
		description="This program handles data collection and user enrollment for real-time face recognition",
		epilog="Warning! Choosing an already used ID will overwrite over the already enrolled subject"
	)
    
    parser.add_argument('-ic', '--image-count', dest="image_count", type=int, default=200, help="Amount of images taken for each subject. Default is 500")
    arguments = parser.parse_args()

    # Requirements
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        content = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        open("haarcascade_frontalface_default.xml", "wb").write(content.content)
    
    if arguments.image_count < 10:
        print("Please choose at least 10 images for each subject")
        exit(-1)
    
    if arguments.image_count > 1000:
        print(f"You have chosen to take {arguments.image_count} images for each subject. Such a large number could cause slow downs!")
        _ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
        if _ != '': exit(-1)

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    id = input("Enter Your ID: ")
    name = input("Enter Your Name: ")
    count = 0
    for _ in range(arguments.image_count):
        ret, frame = video.read()
        gray_iamge = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray_iamge, 1.3, 5)
        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f'datasets/User.{id}.{count}.{name}.jpg', gray_iamge[y:y+h, x:x+w])
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        
        cv2.imshow("Saved frame", frame)
        cv2.waitKey(1)
    
    video.release()
    cv2.destroyAllWindows()
    print("--------------- Image Collection Completed ---------------")

if __name__ == "__main__":
    main()