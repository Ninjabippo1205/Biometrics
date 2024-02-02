import cv2
import iris_recognition as IR
from scipy.spatial import distance as scipydistance
import multiprocessing
                
def hammingdistance(probe, galleryimage, queue, gallerysubject):
	galleryimage = IR.getTemplate(galleryimage).ravel()
	queue.put((scipydistance.hamming(probe, galleryimage), gallerysubject))
	return 0 #correct exit code

def euclideanDistance(probe, galleryimage, queue, gallerysubject):
	galleryimage = IR.getTemplate(galleryimage).ravel()
	queue.put((scipydistance.euclidean(probe.flatten(), galleryimage.flatten()), gallerysubject))
	return 0

def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold, subprocess_count=7):
	minDistance = float("inf")
	matched = ''
	matched_list = {}

	#for image_path in probe:
	image_path = probe
	probeimage = cv2.imread(f"{path}/{test_subject}/{image_path}")
	probeimage = IR.getTemplate(probeimage).ravel()

	jobs = []
	distances = []
	queue = multiprocessing.Queue()

	for gallery_subject in range(len(gallery)):
		distances.append(list())
		for i in range(len(gallery[gallery_subject])):
			test_path = gallery[gallery_subject][i]
			if(image_path == test_path): continue

			galleryimage = cv2.imread(f'{path}/{gallery_subjects[gallery_subject]}/{test_path}')
			p = multiprocessing.Process(target=euclideanDistance, args=(probeimage, galleryimage, queue, gallery_subject, ))
			jobs.append(p)
			p.start()

			if(len(jobs) > subprocess_count):
				for j in jobs: j.join()
				jobs.clear()

				while not queue.empty():
					distance, gallerysub = queue.get()
					distances[gallerysub].append(distance)
	
	for p in jobs: p.join()
	while not queue.empty():
		distance, gallerysub = queue.get()
		distances[gallery_subject].append(distance)

	queue.close()
	queue.join_thread()
	
	for gallery_subject in range(len(distances)):
		for dist in distances[gallery_subject]:

			if(dist < minDistance):
				minDistance = dist
				matched = gallery_subjects[gallery_subject]

			if(dist < threshold):
				matched_list[gallery_subjects[gallery_subject]] = dist

	matched_list = dict(sorted(matched_list.items(), key=lambda item: item[1]))	
	return [matched, minDistance, matched_list]