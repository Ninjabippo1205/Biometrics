import cv2
import math
import numpy as np
import os

# GLOBAL VARIABLES
#####################################
# Holds the pupil's center
centroid = (0,0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = []
#####################################

def getNewEye(list):
	global currentEye
	if (currentEye >= len(list)):
		currentEye = 0
	newEye = list[currentEye]
	currentEye += 1
	return (newEye)

def getIris(frame):
	iris = []
	copyImg = frame.copy()
	resImg = frame.copy()
	grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	mask = np.zeros_like(frame[:,:,0])
	#mask = np.dstack([mask] * 3)
	grayImg = cv2.Canny(grayImg, 5, 70, 3)
	grayImg = cv2.GaussianBlur(grayImg, (7, 7), 0)

	circles = getCircles(grayImg)
	iris.append(resImg)
	for circle in circles:
		rad = int(circle[0][2])
		global radius
		radius = rad
		cv2.circle(mask, (int(centroid[0]), int(centroid[1])), int(rad), (255, 255, 255), thickness=cv2.FILLED) 
		#mask = cv2.bitwise_not(mask)
		resImg = cv2.bitwise_and(frame, copyImg, mask=mask)
		x = int(centroid[0] - rad)
		y = int(centroid[1] - rad)
		w = int(rad * 2)
		h = w
		x = max(0, x)
		y = max(0, y)
		w = min(resImg.shape[1] - x, w)
		h = min(resImg.shape[0] - y, h)
		resImg = resImg[y:y+h, x:x+w]
		cropImg = np.zeros((h, w, 3), dtype=np.uint8)
		cropImg = resImg.copy()		
		return(cropImg)

	return (resImg)

def getCircles(image):
	i = 200
	j=50
	while i > j:
		circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=2, minDist=100.0,
                           param1=30, param2=i, minRadius=100, maxRadius=140)		
		#circles = np.asarray(storage)
		if(circles is not None):
			if (len(circles) == 1):
				return circles
		i -=1
	return ([])

def getPupil(frame):
	height, width = frame.shape[:2] 
	pupilImg = np.zeros((height, width), dtype=np.uint8) 
	lower_bound = np.array([30, 30, 30], dtype=np.uint8) 
	upper_bound = np.array([80, 80, 80], dtype=np.uint8) 
	pupilImg = cv2.inRange(frame, lower_bound, upper_bound) 
	contours, _ = cv2.findContours(pupilImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
	del pupilImg 
	pupilImg = frame.copy() 
	for contour in contours: 
		moments = cv2.moments(contour) 
		area = moments['m00'] 
		if (area > 50): 
			x = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0 
			y = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0 
			pupil = contour 
			global centroid 
			centroid = (int(x),int(y)) 
			cv2.drawContours(pupilImg, pupil, -1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA) 
			cv2.fillPoly(pupilImg, pts =[pupil], color=(0,0,0))
			break 
	return (pupilImg)

def get_polar_to_cart_img(image, rad): 
	img_size = image.shape
	c = (float(img_size[1]/2.0), float(img_size[0]/2.0))
	img_res=cv2.logPolar(image, c, img_size[1]/math.log(rad), cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)
	return img_res 

#######################################
def gaborconvolve_f(img, minw_length, mult, sigma_f):
    # Convert the image to grayscale if it is a color image
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	img_size = img.shape
	rows= img_size[0]
	ndata = img_size[1]
	logGabor_f = np.zeros(ndata)
	filterb = np.zeros([rows, ndata], dtype=complex)
	
	radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
	radius[0] = 1

    # filter wavelength
	wavelength = minw_length

    # radial filter component 
	fo = 1 / wavelength
	logGabor_f[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigma_f)**2))
	logGabor_f[0] = 0
    # convolution for each row
	for r in range(rows):
		signal = img[r, 0:ndata]
		imagefft = np.fft.fft(signal)
		filterb[r, :] = np.fft.ifft(imagefft * np.conj(logGabor_f)).real
	
	return filterb




def encode_iris(arr_polar, minw_length, mult, sigma_f):
    # convolve with gabor filters
    filterb = gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
    l = arr_polar.shape[1]
    template = np.zeros([arr_polar.shape[0], 2 * l])

    # making the iris template
    filt = filterb[:, :]

    # quantization and check to se if the phase data is useful
    H1 = np.real(filt) > 0
    H2 = np.imag(filt) > 0

    H3 = np.abs(filt) < 0.0001
    for i in range(l):
        ja = 2 * i

        # biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]

    return template
#######################################
 
# Window creation for showing input, output 
cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE) 
cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE) 
cv2.namedWindow("normalized", cv2.WINDOW_AUTOSIZE)

eyes_list = os.listdir('images') 
key = 0
while len(eyes_list) > 0: 
	eye = getNewEye(eyes_list) 
	eyes_list.remove(eye)
	frame = cv2.imread("images/" + eye) 
	iris = frame.copy() 
	output = getPupil(frame) 
	iris = getIris(output) 
	cv2.imshow("input", frame) 
	cv2.imshow("output", iris) 
	norm_img = iris.copy()
	norm_img = get_polar_to_cart_img(iris, radius) 
	cv2.imshow("normalized", norm_img) 
	##############################################
	#minw_length = 18
	#mult = 1
	#sigma_f = 0.5
	#template = encode_iris(norm_img, minw_length, mult, sigma_f)
	#print(template.shape)
	#path = str(eye[:-4]+".txt")
	#np.savetxt(path, template)
	##############################################
	key = cv2.waitKey(3000) 
	if key == 27 or key == 1048603: 
		break 
 
cv2.destroyAllWindows()