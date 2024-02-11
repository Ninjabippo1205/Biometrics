import cv2, os, numpy as np
from scipy.spatial import distance as scipydistance

import iris_processing as IrisProcessing

def subject_euclidean_distance(path, gallery_subject, probeimage, probe_path):
    sum = 0; inc = 0

    images = []
    if os.path.exists(f"{path}/{gallery_subject}/L"):
        images += ["L/" + img for img in os.listdir(f"{path}/{gallery_subject}/L")]
    if os.path.exists(f"{path}/{gallery_subject}/R"):
        images += ["R/" + img for img in os.listdir(f"{path}/{gallery_subject}/R")]
    
    for img in images:
        img_path = f"{path}/{gallery_subject}/{img}"

        if img_path == probe_path:
            inc = 1
            continue

        if os.path.exists(f'template/{gallery_subject}/{img[:-4]}.npy'):
            galleryimage = np.load(f'template/{gallery_subject}/{img[:-4]}.npy')
        else:
            galleryimage = cv2.imread(img_path)
            galleryimage = IrisProcessing.getTemplate(galleryimage)
            IrisProcessing.saveTemplate(galleryimage, f'template/{gallery_subject}/{img[:-4]}.npy')

        sum += scipydistance.euclidean(probeimage, galleryimage)
        
    return [gallery_subject, sum/(len(images) - inc)]  # Return distances instead of putting it in a queue

def image_matching(path, probe_path, gallery_subjects, threshold, process_pool):
    probeimage = cv2.imread(probe_path)
    probeimage = IrisProcessing.getTemplate(probeimage)
    
    args = []
    for gallery_subject in gallery_subjects:
        args.append([path, gallery_subject, probeimage, probe_path])

    distances = process_pool.starmap(func=subject_euclidean_distance, iterable=args)

    sorted_distances = sorted(distances, key=lambda x: x[1])
    matched_list = [sublist for sublist in sorted_distances if sublist[1] < threshold]
    return matched_list