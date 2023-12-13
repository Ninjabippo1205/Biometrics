import cv2
import math
import numpy as np
import os
 
dataset = {}
enrolled_users_intervals = [[1,1]]
enrolled_image_intervals = [[1,1], [5,6]]

def createDatasetfromPath(files_in_directory):
		for folder in files_in_directory:
			for interval in enrolled_users_intervals:
				if(int(folder) >= interval[0] and int(folder) <= interval[1]):
					dataset[folder] = {}
					dataset[folder]["L"] = []
					dataset[folder]["R"] = []
					files_L = os.listdir("Biometric Systems/CASIA-Iris-Lamp/" + folder + "/" + "L")
					files_R = os.listdir("Biometric Systems/CASIA-Iris-Lamp/" + folder + "/" + "R")
					counter = 1
					for L in files_L:
						for image_interval in enrolled_image_intervals:
							if(counter >= image_interval[0] and counter <= image_interval[1]):
								dataset[folder]["L"].append("Biometric Systems/CASIA-Iris-Lamp/" + folder + "/" + "L" + "/" + L)
						counter += 1
					counter = 1
					for R in files_R:
						for image_interval in enrolled_image_intervals:
							if(counter >= image_interval[0] and counter <= image_interval[1]):
								dataset[folder]["R"].append("Biometric Systems/CASIA-Iris-Lamp/" + folder + "/" + "R" + "/" + R)	
						counter += 1

def viewImages(dataset, timeout):
	for i in dataset:
		for j in dataset[i]:
			for k in dataset[i][j]:
				frame = cv2.imread(k)
				cv2.imshow("input", frame)
				key = cv2.waitKey(timeout)
				if key == 27 or key == 1048603:
					break



files_in_directory = os.listdir("Biometric Systems/CASIA-Iris-Lamp")
createDatasetfromPath(files_in_directory)

#viewImages(dataset, 100)

def getCircles(originale, image):
	origin = originale.copy()
	img = image.copy()
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Trova i cerchi con la trasformata di Hough per i cerchi
	circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=2, minDist=20, param1=50, param2=200, minRadius=0, maxRadius=0)

	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0, :]:
			# Disegna il cerchio esterno
			cv2.circle(origin, (i[0], i[1]), i[2], (0, 255, 0), 2)
			cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
			# Disegna il centro del cerchio
			cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)
			cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
	return origin, img
		

def preProcessing_image(image, threshold_value, blur):
	_, edges = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
	sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
	# Calcola la magnitudine complessiva
	sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

	## Normalizza i valori tra 0 e 255 per la visualizzazione
	sobel_magnitude_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)

	# Converti i risultati in formato uint8
	sobel_magnitude_normalized_uint8 = np.uint8(sobel_magnitude_normalized)
	res = cv2.GaussianBlur(sobel_magnitude_normalized_uint8, blur, 0)

	return res

for i in dataset:
	for j in dataset[i]:
		for k in dataset[i][j]:
			frame = cv2.imread(k)
			cv2.imshow("frame", frame)
			preProcessed_image = preProcessing_image(frame, threshold_value=110, blur=(15,15))
			circles, image_with_circles = getCircles(frame, preProcessed_image)
			cv2.imshow("preProcessed image with circles", image_with_circles)
			cv2.imshow("circles", circles)
			key = cv2.waitKey(6000)
			if key == 27 or key == 1048603:
				break




cv2.destroyAllWindows()