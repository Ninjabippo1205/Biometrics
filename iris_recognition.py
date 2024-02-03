# project imports
import iris_identification
import iris_processing

# global imports
import cv2, os, random
import numpy as np

def createDatasetfromPath(path):
	dataset = {}
	for folder in os.listdir(path):
		# Adding left eye images
		if os.path.exists(f"CASIA-Iris-Lamp/{folder}/L"): # (as long as it exists)
			dataset[f"{folder}/L"] = os.listdir("CASIA-Iris-Lamp/" + folder + "/" + "L")

		# Adding right eye images
		if os.path.exists(f"CASIA-Iris-Lamp/{folder}/R"):
			dataset[f"{folder}/R"] = os.listdir("CASIA-Iris-Lamp/" + folder + "/" + "R")
	return dataset

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

def getCircles(image, param1, param2, minDist, minRadius, maxRadius):
	img = image.copy()

	circles = None
	while circles is None:
		circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=2, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
		param2 -= 1
	return circles

def getPupil(image, param1, param2, minDist, minRadius, maxRadius):
	img = image.copy()
	circles = np.uint16(np.around(getCircles(img, param1, param2, minDist, minRadius, maxRadius)))
	pupil = None
	for i in circles[0, :]:
		centro = (i[0], i[1])
		raggio = i[2]
		cv2.circle(img, centro, raggio, (0, 255, 0), 2)
		cv2.circle(img, centro, 2, (0, 0, 255), 3)

		pupil = np.zeros_like(image)
		cv2.circle(pupil, centro, raggio, (255,255,255), -1)

		return centro, raggio
	return (0,0), 0

def getIris(image, param1, param2, minDist, minRadius, maxRadius):
	img = image.copy()
	circles = np.uint16(np.around(getCircles(img, param1, param2, minDist, minRadius, maxRadius)))
	for i in circles[0, :]:
		cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

		return (i[0], i[1]), i[2]
	return (0,0), 0

def preProcessing_iris_image(img):
	img = cv2.GaussianBlur(img, (27, 27), 0)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1) #15, 0.5
	img = cv2.GaussianBlur(img, (27, 27), 0)
	edges = cv2.Canny(img, 70, 120)#50,150
	img = cv2.GaussianBlur(edges, (27, 27), 0)
	return img

def preProcessing_pupil_image(img, blur, threshold_value, sobel):
	image = img.copy()
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
	# Crea una griglia di angoli e raggi
	radii = np.linspace(pupil_radius, iris_radius, 100)  	# altezza proporzionale alla differenza dei raggi
	angles = np.arange(0, 2*np.pi, 2*np.pi/360)						# larghezza proporzionale alla circonferenza dell'iride
	
	unwrapped_img = np.zeros((radii.size, angles.size))

	for i in range(radii.size):
		for j in range(angles.size):
			unwrapped_img[i, j] = img[int(center[1] + radii[i]*np.sin(angles[j])), int(center[0] + radii[i]*np.cos(angles[j]))]
	return unwrapped_img

def eyelid_mask_after_normalization(img):
	kernel = np.ones((8,8),np.uint8)
	img_eroded = cv2.erode(img, kernel, iterations = 1)
	contorni, hierarchy = cv2.findContours(img_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contorni = [contour for contour, h in zip(contorni, hierarchy[0]) if h[3] == -1]
	max_area = max(cv2.contourArea(contorno) for contorno in contorni)
	new = np.zeros_like(img)
	for contorno in contorni:
		area_contorno = cv2.contourArea(contorno)
		if area_contorno < 1000:
			continue
		if area_contorno < max_area/6:
			continue
		cv2.drawContours(new, [contorno], -1, 255, -1)
	return new

def eyelid_mask_before_normalization(img, pupil_center, pupil_radius, iris_center, iris_radius):
	blurred = cv2.GaussianBlur(img, (15, 15), 0) #21,21

	mask = np.zeros_like(img)
	cv2.circle(mask, iris_center, iris_radius, (255,255,255), -1)
	cv2.circle(mask, pupil_center, pupil_radius, (0,0,0), -1)
	mean = cv2.mean(blurred, mask=mask)[0]

	if mean > 120:
		thresh = cv2.threshold(blurred, 0.43*mean, 255, cv2.THRESH_TOZERO)[1]
		thresh = cv2.threshold(thresh, 1*mean, 255, cv2.THRESH_TOZERO_INV)[1]
	elif mean > 100:
		thresh = cv2.threshold(blurred, 0.7*mean, 255, cv2.THRESH_TOZERO)[1]
		thresh = cv2.threshold(thresh, 1.35*mean, 255, cv2.THRESH_TOZERO_INV)[1]
	elif mean > 80:
		thresh = cv2.threshold(blurred, 0.66*mean, 255, cv2.THRESH_TOZERO)[1]
		thresh = cv2.threshold(thresh, 1.36*mean, 255, cv2.THRESH_TOZERO_INV)[1]
	else:
		thresh = cv2.threshold(blurred, 0.5*mean, 255, cv2.THRESH_TOZERO)[1]
		thresh = cv2.threshold(thresh, 1.4*mean, 255, cv2.THRESH_TOZERO_INV)[1]
	
	mask = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)[1]
	
	contorni, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contorni = [contour for contour, h in zip(contorni, hierarchy[0]) if h[3] == -1]
	new = np.zeros_like(img)

	max_area = max(cv2.contourArea(contorno) for contorno in contorni)
	c = 0
	avg_color_contour = []
	for contorno in contorni:
		if cv2.contourArea(contorno) < max_area/4:
			continue
		contour_mask = np.zeros_like(img)
		cv2.drawContours(contour_mask, [contorno], -1, 255, 2)
		cv2.drawContours(new, [contorno], -1, 255, 2)
		c = c+1
		avg_color_contour.append([cv2.mean(img, mask=contour_mask)[0], contorno])

	contorno = sorted(avg_color_contour, key=lambda x: x[0], reverse=True)[0][1]
	res = np.zeros_like(img)
	cv2.drawContours(res, [contorno], -1, 255, -1)
	ris = cv2.bitwise_and(res, mask)
	return ris

def getTemplate(path):
	frame = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
	preProcessed_image_for_pupil = preProcessing_pupil_image(frame, threshold_value=60, blur=(15,15), sobel=3)
	pupil_center, pupil_radius = getPupil(preProcessed_image_for_pupil, param1=50, param2=400, minDist=0.5, minRadius=15, maxRadius=70)
			
	preProcessed_image_for_iris = preProcessing_iris_image(frame)
	min_radius = int(pupil_radius + (0.3*pupil_radius))
	max_radius = int(pupil_radius + (0.2*pupil_radius) + 100)	
	iris_center, iris_radius = getIris(preProcessed_image_for_iris, param1=30, param2=400, minDist=0.01, minRadius=min_radius, maxRadius=max_radius)

	lashes_mask = eyelashes_mask(frame.copy())
	partial_iris = not_iris_mask(pupil_center, iris_center, pupil_radius, iris_radius, frame)
			
	p_iris = eyelid_mask_before_normalization(partial_iris, pupil_center, pupil_radius, iris_center, iris_radius)
	partial_iris = cv2.bitwise_and(partial_iris, partial_iris, mask=p_iris)
	iris_without_eyelid_and_eyelashes_before_normalization = cv2.bitwise_and(partial_iris, lashes_mask)

	# Adding eyelashes mask to image and normalizing
	normalized_iris = normalizeWithPolarCoordinates(iris_without_eyelid_and_eyelashes_before_normalization, iris_center, pupil_radius, iris_radius)
	normalized_iris = normalized_iris.astype('uint8')
	
	eyelid_mask_ = eyelid_mask_after_normalization(normalized_iris)
	final_iris = cv2.bitwise_and(normalized_iris, normalized_iris, mask=eyelid_mask_)

	filters = build_filters()
	filitered_iris = process(final_iris, filters)

	#vector = binarize(filitered_iris)

	return filitered_iris

def process(img, filters):
	accum = np.zeros_like(img)
	for kern, params in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum

def binarize(img, threshold=128):
	return (img > threshold).astype(np.uint8)

def build_filters():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi/16):
		params = {'ksize' : (ksize, ksize), 'sigma' : 1.0, 'theta' : theta, 'lambd' : 15.0, 'gamma' : 2, 'psi' : 0, 'ktype' : cv2.CV_32F}
		kern = cv2.getGaborKernel(**params)
		kern /= 1.5 * kern.sum()
		filters.append((kern, params))
	
	return filters


def main():
	path = "CASIA-Iris-Lamp"
	threshold = 7300

	# Creating dataset
	dataset = createDatasetfromPath(path=path)
	d_keys = list(dataset.keys()); random.shuffle(d_keys)
	
	# Get a test subject based on the shuffled keys
	test_subject = d_keys[random.randint(0, len(d_keys)-1)]
	probe = dataset[test_subject] # Probe is a list of images

	# Using the first 20 elements to use as gallery
	gallery_subjects = d_keys[:7]
	# Checking that there is both left and right eye for every subject
	for eye in gallery_subjects:
										# 158 is ascii for L+R. By removing a letter, the other ascii number will pop up
		if not eye[:-1]+chr(158 - ord(eye[-1])) in gallery_subjects: gallery_subjects.append(eye[:-1]+chr(158 - ord(eye[-1])))


	gallery = {} # Gallery is a subset of the dictionary "dataset"
	for x in gallery_subjects: gallery[x] = dataset[x]

	# Calculating False Acceptance, Good Rejection, Detect Indentification, Total Genuine and Total Impostor
	#									(yes|no),				(no|no),						(yes|yes),						(), 								()  
	FA = 0; GR = 0; TG =len(gallery_subjects); TI = len(gallery_subjects)
	DI = np.zeros(len(gallery_subjects*40)-1)

	## image matching ##
	for test_subject in d_keys:
		probe = dataset[test_subject][0]
	
		subject_matched, minimum_distance, matched_list = iris_identification.image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold)
		find = False

		if(minimum_distance < threshold):
			if(subject_matched == test_subject):
				DI[0] = DI[0] + 1
				for m in matched_list.keys():
					if m != test_subject:
						FA = FA + 1
						find = True
						break

				if find == False:
					GR = GR +1	
			else:
				i = 0
				for m in matched_list.keys():
					if m == test_subject and i != 0:
						DI[i] = DI[i] +1
						break

					i = i + 1
				FA = FA + 1
				

			print(f"The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects}")
			os.system(f"echo The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects} >> result.txt")
		else:
			GR = GR + 1
			print(f"Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects}")
			os.system(f"echo Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects} >> result.txt")

		
	DIR = np.zeros(len(gallery_subjects)-1)
	DIR[0]= DI[0]/TG
	FRR = 1 - DIR[0]
	FAR = FA / TI
	GRR = GR / TI

	for i in range (1, len(gallery_subjects)-1):
		DIR[i]= DI[i]/(TG+ DIR[i-1])

	print("performance evaluation:", DIR, FRR, FAR, GRR)

if __name__ == "__main__":
	main()