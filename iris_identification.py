import cv2, numpy as np
from scipy.spatial import distance as scipydistance
import os

import iris_processing as IrisProcessing

#def subject_euclidean_distance(gallery, gallery_subject, probeimage, path, image_path):
#    distances = []
#
#    for i in range(len(gallery[gallery_subject])):
#        test_path = gallery[gallery_subject][i]
#        if(image_path == test_path): continue
#
#        try:
#            #raise FileNotFoundError
#            galleryimage = np.load(f'template/{gallery_subject}/{test_path[:-4]}.npy')
#        except FileNotFoundError:
#            galleryimage = cv2.imread(f'{path}/{gallery_subject}/{test_path}')
#            galleryimage = IrisProcessing.getTemplate(galleryimage).flatten()
#            IrisProcessing.saveTemplate(galleryimage, f'template/{gallery_subject}/{test_path[:-4]}.npy')
#
#        #galleryimage_normalized = (galleryimage - np.mean(galleryimage, axis=0)) / np.std(galleryimage, axis=0)
#        #probeimage_normalized = (probeimage - np.mean(probeimage, axis=0)) / np.std(probeimage, axis=0)
#        distances.append(scipydistance.euclidean(probeimage, galleryimage))
#
#    return distances  # Return distances instead of putting it in a queue

def subject_euclidean_distance2(gallery_subject, probeimage, probe_path):
    images = []
    templates = []
    sum = 0

    if os.path.exists(f"CASIA-Iris-Interval/{gallery_subject}/L"):
        images = os.listdir(f"CASIA-Iris-Interval/{gallery_subject}/L")
    if os.path.exists(f"CASIA-Iris-Interval/{gallery_subject}/R"):
        images = images + os.listdir(f"CASIA-Iris-Interval/{gallery_subject}/R")
    
    for img in images:
        
        L_or_R = ""
        if "L" in img: L_or_R = "L"
        if "R" in img: L_or_R = "R"
        img_path = f"CASIA-Iris-Interval/{gallery_subject}/{L_or_R}/{img}"
        if img_path != probe_path:
            try:
                #raise FileNotFoundError
                galleryimage = np.load(f'template/{gallery_subject}/{L_or_R}/{img[:-4]}.npy')
                if galleryimage is None: raise FileNotFoundError
                templates.append(galleryimage)
            except FileNotFoundError:
                galleryimage = cv2.imread(img_path)
                galleryimage = IrisProcessing.getTemplate(galleryimage)#.flatten()
                IrisProcessing.saveTemplate(galleryimage, f'template/{gallery_subject}/{L_or_R}/{img[:-4]}.npy')
                templates.append(galleryimage)
    
    for template in templates:
        sum = sum + scipydistance.euclidean(probeimage, template)
        
    return [gallery_subject, sum/len(templates)]  # Return distances instead of putting it in a queue

def image_matching(path, test_subject, probe_path, gallery_subjects, threshold, process_pool):
    probeimage = cv2.imread(probe_path)
    probeimage = IrisProcessing.getTemplate(probeimage)#.flatten()
    args = []
    for gallery_subject in gallery_subjects:
        args.append([gallery_subject, probeimage, probe_path])

    distances = process_pool.starmap(subject_euclidean_distance2, args)

    sorted_distances = sorted(distances, key=lambda x: x[1])
    matched_list = [sottolista for sottolista in sorted_distances if sottolista[1] < threshold]
    if len(matched_list) == 0:
        return matched_list
    return matched_list

    # Finding minimum distance based on precalculated distances
#    for gallery_subject in range(len(distances)):
#        for dist in distances[gallery_subject]:
#
#            if(dist < minDistance):
#                minDistance = dist
#                matched = gallery_subjects[gallery_subject]
#
#            if(dist < threshold):
#                matched_list[gallery_subjects[gallery_subject]] = dist
#
#    matched_list = dict(sorted(matched_list.items(), key=lambda item: item[1]))    
#    return [matched, minDistance, matched_list]

