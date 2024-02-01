import cv2
import iris_recognition as IR
from scipy.spatial import distance as scipydistance


import numpy as np

def pad_vector(vector, target_length):
    padding_length = target_length - len(vector)
    if padding_length > 0:
        vector = np.pad(vector, (0, padding_length), 'constant')
    return vector



def hammingdistance(img1, img2):
	template1 = IR.getTemplate(img1)
	template2 = IR.getTemplate(img2)

	if(len(template1)>len(template2)):
		template2=pad_vector(template2,len(template1))
	else:
		template1=pad_vector(template1,len(template2))


	#w1, h1 = template1.shape
	#w2, h2 = template2.shape

	#w = int((w1+w2)/2)
	#h = int((h1+h2)/2)
	
	#template1 = cv2.resize(template1, (w, h))
	#template2 = cv2.resize(template2, (w, h))

	#cv2.imshow("feature1", template1) 
	#cv2.imshow("feature2", template2) 
	#key = cv2.waitKey(3000)
	#if(key==27 or key == 1048603):
	#	return template1
	return scipydistance.hamming(template1.ravel(), template2.ravel())


def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold):
	minDistance = float("inf")
	matched = ''
	matched_list = {}
		
	#for image_path in probe:
	image_path = probe
	probeimage = cv2.imread(f"{path}/{test_subject}/{image_path}")
	
	for gallery_subject in range(len(gallery)):
		for test_path in gallery[gallery_subject]:

			if(image_path == test_path): continue

			galleryimage = cv2.imread(f"{path}/{gallery_subjects[gallery_subject]}/{test_path}")
			
			d = hammingdistance(probeimage, galleryimage)
			if(d < minDistance):
				minDistance = d
				matched = gallery_subjects[gallery_subject]

			if(d < threshold):
				matched_list[gallery_subjects[gallery_subject]] = d

	matched_list = dict(sorted(matched_list.items(), key=lambda item: item[1]))	
	return [matched, minDistance, matched_list]