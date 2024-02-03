import cv2
import iris_recognition as IR
from scipy.spatial import distance as scipydistance
from multiprocessing import Pool

                
def subject_hamming_distance(gallery, gallery_subjects, gallery_subject, path, image_path, probeimage):
    distances = []

    for i in range(len(gallery[gallery_subject])):
        test_path = gallery[gallery_subject][i]
        if(image_path == test_path): continue

        galleryimage = cv2.imread(f'{path}/{gallery_subjects[gallery_subject]}/{test_path}')
        galleryimage = IR.getTemplate(galleryimage).ravel()

        distances.append(scipydistance.hamming(probeimage, galleryimage))
    
    return distances  # Return distances instead of putting it in a queue


def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold, subprocess_count=14):
    minDistance = float("inf")
    matched = ''
    matched_list = {}

    # Getting template for probe
    probeimage = cv2.imread(f"{path}/{test_subject}/{probe}")
    probeimage = IR.getTemplate(probeimage).ravel()

    # Create a pool of processes
    with Pool(processes=subprocess_count) as pool:
    # Use a process for each subject
        results = []
        for gallery_subject in range(len(gallery)):
            result = pool.apply_async(subject_hamming_distance, args=(gallery, gallery_subjects, gallery_subject, path, probe, probeimage))
            results.append(result)

    	# Collect the results as they become available
        distances = [result.get() for result in results]

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