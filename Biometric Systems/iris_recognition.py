import cv2
import math
import numpy as np
import os

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


#def getCircles(originale, image, param1=50, param2=300, minDist=20):
#	origin = originale.copy()
#	img = image.copy()
#	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#	circles = None
#	while circles is None:
#		circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=2, minDist=minDist, param1=param1, param2=param2, minRadius=0, maxRadius=0)
#		param2 -= 1
#
#	if circles is not None:
#		circles = np.uint16(np.around(circles))
#		for i in circles[0, :]:
#			# Disegna il cerchio esterno
#			cv2.circle(origin, (i[0], i[1]), i[2], (0, 255, 0), 2)
#			cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#			# Disegna il centro del cerchio
#			cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)
#			cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#	return origin, img

def getCircles(image, param1, param2, minDist, minRadius, maxRadius):
	img = image.copy()
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	circles = None
	while circles is None:
		circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=2, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
		param2 -= 1
	return circles

def getPupil(original, image, param1=50, param2=300, minDist=20, minRadius=0, maxRadius=0):
	img = image.copy()
	origin = original.copy()
	circles = np.uint16(np.around(getCircles(img, param1, param2, minDist, minRadius, maxRadius)))
	pupil = None
	iris = None
	for i in circles[0, :]:
		cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

		cv2.circle(origin, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)

		#cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), -1)
		#cv2.circle(origin, (i[0], i[1]), i[2], (0, 0, 255), -1)
		
		##da cancellare roba sotto

		#dentro = int(i[2]+20)
		#fuori = int(i[2]+100)

		#cv2.circle(img, (i[0], i[1]), (dentro), (255, 255, 0), 2)
		#cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
		#cv2.circle(img, (i[0], i[1]), (fuori), (255, 0, 255), 2)
		#cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

		#cv2.circle(origin, (i[0], i[1]), (dentro), (255, 255, 0), 2)
		#cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)
		#cv2.circle(origin, (i[0], i[1]), (fuori), (255, 0, 255), 2)
		#cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)

		##fine da cancellare

		pupil = np.zeros_like(image)
		cv2.circle(pupil, (i[0], i[1]), i[2], (255,255,255), -1)
		iris = np.zeros_like(image)
		cv2.circle(iris, (i[0], i[1]), i[2]+100, (255,255,255), -1)

		return origin, img, pupil, iris,  i[2]
	return origin, img, pupil, iris, 0

def getIris(original, image, param1=50, param2=300, minDist=20, minRadius=0, maxRadius=0):
	img = image.copy()
	origin = original.copy()
	circles = np.uint16(np.around(getCircles(img, param1, param2, minDist, minRadius, maxRadius)))
	for i in circles[0, :]:
		cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

		cv2.circle(origin, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)

		#cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), -1)
		#cv2.circle(origin, (i[0], i[1]), i[2], (0, 0, 255), -1)

		return origin, img
	return origin, img


def preProcessing__pupil_image(img, threshold_value, blur, mask):
	image = img.copy()

	if mask is not None:
		# Applica la maschera
		image = cv2.add(image, mask)

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

def preProcessing_iris_image(img, threshold_value, blur, pupil_mask, iris_mask):
	image = img.copy()

	if pupil_mask is not None:
		# Applica la maschera
		image = cv2.add(image, pupil_mask)
		

	_, edges = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
	sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
	# Calcola la magnitudine complessiva
	sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

	## Normalizza i valori tra 0 e 255 per la visualizzazione
	sobel_magnitude_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)

	# Converti i risultati in formato uint8
	sobel_magnitude_normalized_uint8 = np.uint8(sobel_magnitude_normalized)

	res = cv2.GaussianBlur(sobel_magnitude_normalized_uint8, blur, 0)

	return res


dataset = {}
enrolled_users_intervals = [[1,30]]
enrolled_image_intervals = [[1,3], [5,8], [15,20]]

files_in_directory = os.listdir("Biometric Systems/CASIA-Iris-Lamp")
createDatasetfromPath(files_in_directory)
#viewImages(dataset, 100)

for i in dataset:
	for j in dataset[i]:
		for k in dataset[i][j]:
			frame = cv2.imread(k)
			cv2.imshow("frame", frame)
			preProcessed_image_for_pupil = preProcessing__pupil_image(frame, threshold_value=60, blur=(15,15), mask=None)	#70
			origin_pupil, pupil, pupil_mask, iris_mask, pupil_radius = getPupil(frame, preProcessed_image_for_pupil, param1=50, param2=400, minDist=0.5, minRadius=15, maxRadius=70)
			#cv2.imshow("pupilla", pupil)
			#cv2.imshow("origin_pupil", origin_pupil)
			preProcessed_image_for_iris = preProcessing_iris_image(frame, threshold_value=107, blur=(27,27), pupil_mask=pupil_mask, iris_mask=iris_mask)	#110
			cv2.imshow("preProcessed_image_for_iris", preProcessed_image_for_iris)
			origin_iris, iris = getIris(frame, preProcessed_image_for_iris, param1=60, param2=400, minDist=0.1, minRadius=(pupil_radius + 20), maxRadius=(pupil_radius + 100))
			cv2.imshow("iris", iris)
			cv2.imshow("origin_pupil", origin_iris)
			key = cv2.waitKey(1000)
			if key == 27 or key == 1048603:
				break




cv2.destroyAllWindows()