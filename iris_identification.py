import cv2, multiprocessing
import iris_processing as IrisProcess
from scipy.spatial import distance as scipydistance

def subject_euclidean_distance(gallery, gallery_subject, probeimage, path, image_path):
    distances = []

    for i in range(len(gallery[gallery_subject])):
        test_path = gallery[gallery_subject][i]
        if(image_path == test_path): continue
        
        galleryimage = cv2.imread(f'{path}/{gallery_subject}/{test_path}')
        galleryimage = IrisProcess.getTemplate(galleryimage).ravel()
        distances.append(scipydistance.euclidean(probeimage.flatten(), galleryimage.flatten()))

    return distances  # Return distances instead of putting it in a queue

def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold, subprocess_count=multiprocessing.cpu_count() - 4):
    minDistance = float("inf")
    matched = ''
    matched_list = {}

    # Getting template for probe
    probeimage = cv2.imread(f"{path}/{test_subject}/{probe}")
    probeimage = IrisProcess.getTemplate(probeimage).ravel()

    pool = multiprocessing.Pool(
        processes=subprocess_count,
        #maxtasksperchild = 1 # Keep commented for maximum performace
    )

    # Creating args
    args = []
    for gallery_subject in gallery_subjects:
        args.append([gallery, gallery_subject, probeimage, path, probe])

    # Mapping to pool
    distances = pool.starmap(subject_euclidean_distance, args)
    pool.close()
    pool.join()

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