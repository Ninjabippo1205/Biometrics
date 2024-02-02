import cv2
import iris_recognition as IR
from scipy.spatial import distance as scipydistance
import multiprocessing
                
def subject_hamming_distance(gallery, gallery_subjects, gallery_subject, path, image_path, probeimage, queue, ):
	distances = []

	for i in range(len(gallery[gallery_subject])):
		test_path = gallery[gallery_subject][i]
		if(image_path == test_path): continue

		galleryimage = cv2.imread(f'{path}/{gallery_subjects[gallery_subject]}/{test_path}')
		galleryimage = IR.getTemplate(galleryimage).ravel()

		distances.append(scipydistance.hamming(probeimage, galleryimage))
	
	queue.put(distances)
	return 0 #correct exit code

def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold, subprocess_count=7):
	minDistance = float("inf")
	matched = ''
	matched_list = {}

	# Getting template for probe
	probeimage = cv2.imread(f"{path}/{test_subject}/{probe}")
	probeimage = IR.getTemplate(probeimage).ravel()

	# Multiprocessing items
	jobs = []
	distances = []
	queue = multiprocessing.Queue()

	# Use a process for each subject
	for gallery_subject in range(len(gallery)):
		p = multiprocessing.Process(target=subject_hamming_distance, args=(gallery, gallery_subjects, gallery_subject, path, probe, probeimage, queue, ))
		jobs.append(p)
		p.start()

		if len(jobs) > subprocess_count:
			for j in jobs: j.join()
			jobs.clear()

			# Get the process list and add it to the distances 
			while not queue.empty():
				distances.append(queue.get())
	
	# Same for all unfinished processes
	for p in jobs: p.join()
	while not queue.empty():
		distances.append(queue.get())

	# Closing queue
	queue.close()
	queue.join_thread()
	
	# Finding minimum distance based on precalculated distances
	for gallery_subject in range(len(distances)):
		for dist in distances[gallery_subject]:

			if(dist < minDistance):
				minDistance = dist
				matched = gallery_subjects[gallery_subject]

			if(dist < threshold):
				matched_list[gallery_subjects[gallery_subject]] = dist

	matched_list = dict(sorted(matched_list.items(), key=lambda item: item[1]))	
	return [matched, minDistance, matched_list]