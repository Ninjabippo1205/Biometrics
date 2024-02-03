import cv2, multiprocessing
import iris_recognition as IR
from scipy.spatial import distance as scipydistance

                
def subject_hamming_distance(gallery, gallery_subjects, gallery_subject, path, image_path, probeimage):
    distances = []

    for i in range(len(gallery[gallery_subject])):
        test_path = gallery[gallery_subject][i]
        if(image_path == test_path): continue

        galleryimage = cv2.imread(f'{path}/{gallery_subjects[gallery_subject]}/{test_path}')
        galleryimage = IR.getTemplate(galleryimage).ravel()

        distances.append(scipydistance.hamming(probeimage, galleryimage))
    
    return distances  # Return distances instead of putting it in a queue


def image_matching(path, test_subject, probe, gallery, gallery_subjects, threshold, subprocess_count=multiprocessing.cpu_count()):
    minDistance = float("inf")
    matched = ''
    matched_list = {}

    # Getting template for probe
    probeimage = cv2.imread(f"{path}/{test_subject}/{probe}")
    probeimage = IR.getTemplate(probeimage).ravel()

    pool = multiprocessing.Pool(
        processes=subprocess_count,
        maxtasksperchild = 2
    )

    # Creating args
    args = []
    for gallery_subject in range(len(gallery)):
        args.append([gallery, gallery_subjects, gallery_subject, path, probe, probeimage])

    # Mapping to pool
    distances = pool.starmap(subject_hamming_distance, args)
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