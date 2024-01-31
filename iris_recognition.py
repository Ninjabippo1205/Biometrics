# project imports
import iris_identification
import iris_processing

# global imports
import cv2, os, random
import numpy as np

# Global varbiales
verbose = False

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

	if(verbose): cv2.imshow(str("Image with min and max Circles"+str(id)), copy)

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

def eyelid_mask(img):
	h, w = img.shape
	new_bigger_img = np.zeros((h+300, w+300), dtype=np.uint8)
	new_bigger_img[0:h, 0:w] = img
	#cv2.imshow("new_bigger_img", new_bigger_img)
	return img



def getTemplate(path):
	global verbose

	frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
	preProcessed_image_for_pupil = preProcessing_pupil_image(frame, threshold_value=60, blur=(15,15), sobel=3)
	pupil_center, pupil_radius = getPupil(preProcessed_image_for_pupil, param1=50, param2=400, minDist=0.5, minRadius=15, maxRadius=70)
			
	preProcessed_image_for_iris = preProcessing_iris_image(frame)
	min_radius = int(pupil_radius + (0.3*pupil_radius))
	max_radius = int(pupil_radius + (0.2*pupil_radius) + 100)	
	iris_center, iris_radius = getIris(preProcessed_image_for_iris, param1=30, param2=400, minDist=0.01, minRadius=min_radius, maxRadius=max_radius)
				
	lashes_mask = eyelashes_mask(frame.copy())
	partial_iris = not_iris_mask(pupil_center, iris_center, pupil_radius, iris_radius, frame)

	# Adding eyelashes mask to image and normalizing
	normalized_iris = normalizeWithPolarCoordinates(cv2.bitwise_and(partial_iris, lashes_mask), iris_center, pupil_radius, iris_radius)
	final_iris = eyelid_mask(normalized_iris)

	if(verbose): cv2.imshow("normalized_iris", final_iris)

	width = final_iris.shape[0]
	height = final_iris.shape[1]
	
	sigma = 6; theta = 0.8; lambda_ = 10; gamma = 2
	gabor_kernel = cv2.getGaborKernel((width, height), sigma, theta, lambda_, gamma)
	filtered_iris = cv2.filter2D(final_iris, cv2.CV_64F, gabor_kernel)
	if(verbose): cv2.imshow("feature", filtered_iris)
	return filtered_iris

def main(*, verb=False):
	global verbose
	verbose = verb

	path = "CASIA-Iris-Lamp"
	threshold = 0.93

	# Creating dataset
	dataset = createDatasetfromPath(path=path)
	d_keys = list(dataset.keys()); random.shuffle(d_keys)
	
	# Get a test subject based on the shuffled keys
	test_subject = d_keys[random.randint(0, len(d_keys)-1)]
	probe = dataset[test_subject] # Probe is a list of images

	# Using the first 20 elements to use as gallery
	gallery_subjects = d_keys[:10]
	gallery = [] # Gallery is a list of list of images
	for x in gallery_subjects: gallery.append(dataset[x])

	# Calculating False Acceptance, Good Rejection, Detect Indentification, Total Genuine and Total Impostor
	#									(yes|no),				(no|no),						(yes|yes),						(), 								()  
	FA = 0; GR = 0; DI = 0; TG = 0; TI = 0

	## image matching ##
	for test_subject in d_keys:
		probe = dataset[test_subject]
		subject_matched, minimum_distance = iris_identification.image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold)
	
		if(minimum_distance < threshold):
			print(f"The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects}")
			os.system(f"echo The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects} >> result.txt")
		else:
			print(f"Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects}")
			os.system(f"echo Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects} >> result.txt")


	if(verbose):
		key = cv2.waitKey(30000)
		if key == 27 or key == 1048603: return
		cv2.destroyAllWindows()

if __name__ == "__main__":
	main()