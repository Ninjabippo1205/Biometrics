import cv2, numpy as np
from scipy.spatial import distance as scipydistance

import iris_processing as IrisProcessing

def subject_euclidean_distance(gallery, gallery_subject, probeimage, path, image_path):
    distances = []

    for i in range(len(gallery[gallery_subject])):
        test_path = gallery[gallery_subject][i]
        if(image_path == test_path): continue
        
        try:
            galleryimage = np.load(f'template/{gallery_subject}/{test_path[:-4]}.npy')
        except FileNotFoundError:
            galleryimage = cv2.imread(f'{path}/{gallery_subject}/{test_path}')
            galleryimage = IrisProcessing.getTemplate(galleryimage).flatten()
            IrisProcessing.saveTemplate(galleryimage, f'template/{gallery_subject}/{test_path[:-4]}.npy')

        distances.append(scipydistance.euclidean(probeimage, galleryimage))

    return distances  # Return distances instead of putting it in a queue

def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold, process_pool):
    minDistance = float("inf")
    matched = ''
    matched_list = {}

    # Getting template for probe.
    probeimage = cv2.imread(f"{path}/{test_subject}/{probe}")
    probeimage = IrisProcessing.getTemplate(probeimage).flatten()

    # Creating args
    args = []
    for gallery_subject in gallery_subjects:
        args.append([gallery, gallery_subject, probeimage, path, probe])

    # Mapping to pool
    distances = process_pool.starmap(subject_euclidean_distance, args)
    process_pool.close()

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