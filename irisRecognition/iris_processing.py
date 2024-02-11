import cv2, os, numpy as np

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
	max_radius = 0
	max_center = (0,0)

	for i in circles[0, :]:
		centro = (i[0], i[1])
		raggio = i[2]
		if raggio > max_radius:
			max_radius = raggio
			max_center = centro
	#return max_center, (max_radius-10)
	return max_center, max_radius

def preProcessing_pupil_image(img, blur, threshold_value):
	image = img.copy()
	_, edges = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)
	img_eroded = cv2.erode(edges, kernel, iterations = 7)
	res = cv2.GaussianBlur(img_eroded, blur, 0)
	return res

def not_iris_mask(pupil_center, iris_center, pupil_radius, iris_radius, frame):
	img = frame.copy()
	cv2.circle(img, pupil_center, pupil_radius, (0, 0, 0), -1)
	mask = np.zeros_like(img)
	cv2.circle(mask, iris_center, iris_radius, (255,255,255), -1)
	img = cv2.bitwise_and(img, mask)
	return img

def normalizeWithPolarCoordinates(image, center, pupil_radius, iris_radius):
	img = image.copy()
	# Crea una griglia di angoli e raggi
	radii = np.linspace(pupil_radius, iris_radius, 100)  	# height proportional to radius difference
	angles = np.arange(0, 2*np.pi, 2*np.pi/400)						# width proportional to iris circumference

	unwrapped_img = np.zeros((radii.size, angles.size))
	for i in range(radii.size):
		for j in range(angles.size):
			#l'errore è qua, nella parte int(center[1] + radii[i]*np.sin(angles[j]))
			x = int(center[1] + radii[i]*np.sin(angles[j]))
			y = int(center[0] + radii[i]*np.cos(angles[j]))
			if (x >= 280):
				x = 279
			if y >= 320:
				y = 319
			unwrapped_img[i, j] = img[x, y]
	return unwrapped_img

def eyelid_and_eyelashes_mask_after_normalization(img):
	mask = np.zeros_like(img)
	mask[:,:] = 255
	mean = cv2.mean(img)[0]
	mask[img < (mean-40)] = 0

	mask2 = np.zeros_like(img)
	mask2[:,:] = 255
	mask2[img > 230] = 0
	mask2 = cv2.erode(mask2, np.ones((3,3), np.uint8), iterations=1)
	
	return cv2.bitwise_and(mask, mask2)

def getTemplate(image):
	frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	preProcessed_image_for_pupil = preProcessing_pupil_image(frame, threshold_value=90, blur=(15,15))
	pupil_center, pupil_radius = getPupil(preProcessed_image_for_pupil, param1=50, param2=400, minDist=0.5, minRadius=15, maxRadius=70)

	iris_radius = int(pupil_radius + 45)	
	iris_center = pupil_center

	partial_iris = not_iris_mask(pupil_center, iris_center, pupil_radius, iris_radius, frame)
	
	normalized_iris = normalizeWithPolarCoordinates(partial_iris, iris_center, pupil_radius, iris_radius)
	normalized_iris = normalized_iris.astype('uint8')

	eyelid_mask_ = eyelid_and_eyelashes_mask_after_normalization(normalized_iris)
	final_iris = cv2.bitwise_and(normalized_iris, normalized_iris, mask=eyelid_mask_)
	final_iris = enhance_img(final_iris)

	#lbp =  calculate_lbp(final_iris)
	hist = calculate_histograms(final_iris)
	key = cv2.waitKey(0)
	if key == 27:
		cv2.destroyAllWindows()
	return hist
	#return feature_extraction_pca(final_iris)

def calculate_histograms(img):
	n, m = 20, 1
	h, w = img.shape
	cell_h, cell_w = h // n, w // m
	histograms = []
	for i in range(n):
		for j in range(m):
			cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
			cell = calculate_lbp(cell)
			hist = cv2.calcHist([cell], [0], None, [256], [0, 256])
			for k in hist:
				histograms.append(k[0])
	return histograms

def enhance_img(img):
	return cv2.equalizeHist(img)

def calculate_lbp(img, P=8, R=1):
    # Inizializza l'immagine LBP
    lbp = np.zeros_like(img)

    # Calcola LBP
    for i in range(R, img.shape[0]-R):
        for j in range(R, img.shape[1]-R):
            # Ottieni il pixel centrale e i suoi vicini
            center = img[i, j]
            neighbors = [img[i+int(R*np.sin(2*np.pi*p/P)), j+int(R*np.cos(2*np.pi*p/P))] for p in range(P)]
            
            # Calcola il LBP
            lbp[i, j] = sum([2**p if neighbors[p] >= center else 0 for p in range(P)])
    return lbp

# Processed image save functions
def saveDataset(dataset, images_folder):
	if not os.path.exists('template'): os.mkdir('template')

	for object in dataset:
		# Creating folder and saving all templates into it
		if not os.path.exists(f'template/{object[:-2]}'): os.mkdir(f'template/{object[:-2]}')
		if not os.path.exists(f'template/{object}'): os.mkdir(f'template/{object}')
		for image in dataset[object]:
			if os.path.exists(f'template/{object}/{image[:-4]}.npy'): continue
			template = getTemplate(cv2.imread(f"{images_folder}/{object}/{image}")).flatten()
			np.save(f"template/{object}/{image[:-4]}", template)

def saveTemplate(template, path):
	items = path.split('/')

	if not os.path.exists(items[0]): os.mkdir(items[0])
	if not os.path.exists(f'{items[0]}/{items[1]}'): os.mkdir(f'{items[0]}/{items[1]}')
	if not os.path.exists(f'{items[0]}/{items[1]}/{items[2]}'): os.mkdir(f'{items[0]}/{items[1]}/{items[2]}')

	np.save(path, template)

## Tested and unused features ##
# Feature extraction with PCA

# def feature_extraction_pca(img):
# 	pca = PCA(n_components=6)
# 	pca.fit(img)
# 	pca_features = pca.transform(img)
# 	return pca_features

# Feature extraction with wavelength

# def feature_extraction(img):
# 	c = pywt.wavedec2(img, 'db2', mode='periodization', level=3)
# 	c[0] /= np.abs(c[0]).max()
# 	for detail_level in range(3):
# 		c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
# 	#arr, slices = pywt.coeffs_to_array(c)
# 	#plt.imshow(arr, cmap=plt.cm.gray)
# 	#plt.show()
# 	res = np.concatenate([np.ndarray.flatten(c1) for sublist in c for c1 in sublist])
# 	return res

# Feature extraction with Gabor

# def process(img, filters):
# 	accum = np.zeros_like(img)
# 	for kern, params in filters:
# 		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
# 		np.maximum(accum, fimg, accum)
# 	return accum

# def build_filters():
# 	filters = []
# 	ksize = 31
# 	for theta in np.arange(0, np.pi, np.pi/16):
# 		params = {'ksize' : (ksize, ksize), 'sigma' : 3.0, 'theta' : theta, 'lambd' : 10.0, 'gamma' : 0.2, 'psi' : 0, 'ktype' : cv2.CV_32F}
# 		kern = cv2.getGaborKernel(**params)
# 		kern /= 1.5 * kern.sum()
# 		filters.append((kern, params))
# 
# 	return filters