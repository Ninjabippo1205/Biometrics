import cv2
import numpy as np
import os
import math

def createDatasetfromPath(dataset, enrolled_users_intervals, enrolled_image_intervals, files_in_directory):
		for folder in files_in_directory:
			for interval in enrolled_users_intervals:
				if(int(folder) >= interval[0] and int(folder) <= interval[1]):
					dataset[folder] = {}
					dataset[folder]["L"] = []
					dataset[folder]["R"] = []
					files_L = os.listdir("CASIA-Iris-Lamp/" + folder + "/" + "L")
					files_R = os.listdir("CASIA-Iris-Lamp/" + folder + "/" + "R")
					counter = 1
					for L in files_L:
						for image_interval in enrolled_image_intervals:
							if(counter >= image_interval[0] and counter <= image_interval[1]):
								dataset[folder]["L"].append("CASIA-Iris-Lamp/" + folder + "/" + "L" + "/" + L)
						counter += 1
					counter = 1
					for R in files_R:
						for image_interval in enrolled_image_intervals:
							if(counter >= image_interval[0] and counter <= image_interval[1]):
								dataset[folder]["R"].append("CASIA-Iris-Lamp/" + folder + "/" + "R" + "/" + R)	
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

def drawCircle(img, id, center, radius_min, radius_max):
	copy = img.copy()
	copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2RGB)
	cv2.circle(copy, center, radius_min, (0, 165, 255), 2)
	cv2.circle(copy, center, radius_max, (203, 192, 255), 2)
	cv2.imshow(str("Image with min and max Circles"+str(id)), copy)

def getCircles(image, param1, param2, minDist, minRadius, maxRadius):
	img = image.copy()

	circles = None
	while circles is None:
		circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=2, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
		param2 -= 1
	return circles

def getPupil(original, image, param1=50, param2=300, minDist=20, minRadius=0, maxRadius=0):
	img = image.copy()
	origin = original.copy()
	circles = np.uint16(np.around(getCircles(img, param1, param2, minDist, minRadius, maxRadius)))
	pupil = None
	for i in circles[0, :]:
		centro = (i[0], i[1])
		raggio = i[2]
		cv2.circle(img, centro, raggio, (0, 255, 0), 2)
		cv2.circle(img, centro, 2, (0, 0, 255), 3)

		cv2.circle(origin, centro, raggio, (0, 255, 0), 2)
		cv2.circle(origin, centro, 2, (0, 0, 255), 3)

		pupil = np.zeros_like(image)
		cv2.circle(pupil, centro, raggio, (255,255,255), -1)

		return origin, img, pupil, centro, raggio
	return origin, img, pupil, (0,0), 0

def getIris(original, image, param1=50, param2=300, minDist=20, minRadius=0, maxRadius=0):
	img = image.copy()
	origin = original.copy()
	circles = np.uint16(np.around(getCircles(img, param1, param2, minDist, minRadius, maxRadius)))
	for i in circles[0, :]:
		cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

		cv2.circle(origin, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(origin, (i[0], i[1]), 2, (0, 0, 255), 3)

		return origin, img, (i[0], i[1]), i[2]
	return origin, img, (0,0), 0

def preProcessing_iris_image(img):
	#image = cv2.equalizeHist(image)	non l'ho usato ma potrebbe servire magari a rendere i contrasti più netti nell'immagine originale

	img = cv2.GaussianBlur(img, (27, 27), 0)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1) #15, 0.5
	img = cv2.GaussianBlur(img, (27, 27), 0)
	edges = cv2.Canny(img, 70, 120)#50,150
	img = cv2.GaussianBlur(edges, (27, 27), 0)
	return img

def preProcessing_pupil_image(img, blur, threshold_value = 110, pupil_mask=None, pupil_center=None, pupil_radius=None, sobel=3):
	image = img.copy()
	if pupil_mask is not None:
		image = cv2.add(image, pupil_mask)
	_, edges = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=sobel)
	sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=sobel)
	sobelmagnitude_norm = np.uint8(cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX))
	res = cv2.GaussianBlur(sobelmagnitude_norm, blur, 0)
	return res

def not_iris_mask(pupil_center, iris_center, pupil_radius, iris_radius, frame):
	img = frame.copy()
	cv2.circle(img, pupil_center, pupil_radius, (0, 0, 0), -1)
	mask = np.zeros_like(img)
	cv2.circle(mask, iris_center, iris_radius, (255,255,255), -1)
	img = cv2.bitwise_and(img, mask)
	return img

def eyelashes_mask(eye_image):

	# Apply Gaussian blur to reduce noise
	blurred = cv2.GaussianBlur(eye_image, (5, 5), 0)
	
	# Perform adaptive thresholding to create a binary mask
	_, binary_mask = cv2.threshold(blurred, 75, 400, cv2.THRESH_BINARY)
	
	# Perform morphological operations to close small gaps in the binary mask
	kernel = np.ones((5, 5), np.uint8)
	closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
	
	# Use the inverse of the eye mask to segment the eyelashes and eyelid
	return closed_mask

def normalizeWithPolarCoordinates(image, center, pupil_radius, iris_radius):
    img = image.copy()
    y, x = np.indices((img.shape[0], img.shape[1]))
    angle_map = np.arctan2(y-center[1], x-center[0])
    radius_map = np.hypot(y-center[1], x-center[0])
    mask = (radius_map >= pupil_radius) & (radius_map <= iris_radius)
    # Crea una griglia di angoli e raggi
    angles = np.linspace(-np.pi, np.pi, int(2*np.pi*iris_radius))  # larghezza proporzionale alla circonferenza dell'iride
    radii = np.linspace(pupil_radius, iris_radius, int(iris_radius-pupil_radius))  # altezza proporzionale alla differenza dei raggi
    grid_angle, grid_radius = np.meshgrid(angles, radii)
    # Mappa gli angoli e i raggi alla x e alla y
    map_x = grid_radius * np.cos(grid_angle) + center[0]
    map_y = grid_radius * np.sin(grid_angle) + center[1]
    # Interpola i pixel dell'immagine originale sulla griglia
    unwrapped_img = cv2.remap(img, map_x.astype('float32'), map_y.astype('float32'), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Converte l'immagine srotolata in uint8
    unwrapped_img = cv2.normalize(unwrapped_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return unwrapped_img

def main():
	dataset = {}
	enrolled_users_intervals = [[1,5], [45,50]]
	enrolled_image_intervals = [[1,3], [5,8]]
	files_in_directory = os.listdir("CASIA-Iris-Lamp")
	createDatasetfromPath(dataset, enrolled_users_intervals, enrolled_image_intervals, files_in_directory)

	for i in dataset:
		for j in dataset[i]:
			for k in dataset[i][j]:
				frame = cv2.imread(k)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				preProcessed_image_for_pupil = preProcessing_pupil_image(frame, threshold_value=60, blur=(15,15))	#70
				origin_pupil, pupil, pupil_mask, pupil_center, pupil_radius = getPupil(frame, preProcessed_image_for_pupil, param1=50, param2=400, minDist=0.5, minRadius=15, maxRadius=70)
			
				preProcessed_image_for_iris = preProcessing_iris_image(frame)
				min_radius = int(pupil_radius + (0.3*pupil_radius))
				max_radius = int(pupil_radius + (0.2*pupil_radius) + 100)	
				origin_iris, iris, iris_center, iris_radius = getIris(frame, preProcessed_image_for_iris, param1=30, param2=400, minDist=0.01, minRadius=min_radius, maxRadius=max_radius)
				
				lashes_mask = eyelashes_mask(frame.copy())
				partial_iris = not_iris_mask(pupil_center, iris_center, pupil_radius, iris_radius, frame)

				# Adding eyelashes mask to image and normalizing
				final_iris = normalizeWithPolarCoordinates(cv2.bitwise_and(partial_iris, lashes_mask), iris_center, pupil_radius, iris_radius)
				cv2.imshow("final_mask", final_iris)

				key = cv2.waitKey(3000)
				if key == 27 or key == 1048603:
					break

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()







