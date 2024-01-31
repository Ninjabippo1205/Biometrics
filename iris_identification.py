import cv2
from scipy.spatial import distance as scipydistance

def hammingdistance(template1, template2):
	return scipydistance.hamming(template1.ravel(), template2.ravel())


def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold):
	minDistance = float("inf")
	matched = ''
		
	for image_path in probe:
		probeimage = cv2.imread(f"{path}/{test_subject}/{image_path}")

		for gallery_subject in range(len(gallery)):
			for test_path in gallery[gallery_subject]:

				if(image_path == test_path): continue

				galleryimage = cv2.imread(f"{path}/{gallery_subjects[gallery_subject]}/{test_path}")

				d = hammingdistance(probeimage, galleryimage)
				if(d < minDistance):
					minDistance = d
					matched = gallery_subjects[gallery_subject]

	return [matched, minDistance]