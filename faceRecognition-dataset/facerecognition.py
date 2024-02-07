import cv2, os, random, argparse, face_recognition
import numpy as np 
import matplotlib.pyplot as plt
 
def processDataset(path, gallery_percentage, unkown_percentage, no_unknown): 
    if not os.path.exists(path): 
        print("Dataset could not be loaded, please check folder location") 
        exit(-1) 
 
    gallery = {} 
    probe = {} 
 
    subjects = os.listdir(path)

    if not no_unknown:
        # Choosing unknown subjects at random
        while len(probe) < int(len(subjects)*unkown_percentage): 
            item = random.randint(0, len(subjects)-1) 
            probe[subjects[item]] = os.listdir(f'{path}/{subjects[item]}') 
            subjects.remove(subjects[item]) 
 
    for subject in subjects: 
        files = os.listdir(f'{path}/{subject}') 
        gallery[subject] = files[:int(len(files)*gallery_percentage)] 
        probe[subject] = files[int(len(files)*gallery_percentage):] 

    return gallery, probe 
 
def main(): 
    known_encodings = []
    known_names = []

    parser = argparse.ArgumentParser(
		prog="FaceRecognition",
		description="This program recognizes faces based on a face recognition dataset",
		epilog="The dataset must be created as such: \"dataset/xxx/xxx_y.bmp\" where \'x\' is the subject's number and \'y\' the numbered photo."
	)
    
    parser.add_argument('-p', '--path', dest="dataset_path", type=str, default="CASIA-FaceV5", help="Path to the folder where the face images are saved")
    parser.add_argument('-gp', '--gallery-percentage', dest="gallery_percentage", default="0.8", type=float, help="Percentage of images for chosen subjects that will be added to the gallery")
    parser.add_argument('-up', '--unkown-percentage', dest="unknown_percentage", default="0.2", type=float, help="Percentage of unkown subjects that are going to be tested" )
    parser.add_argument('-nu', '--no-unknown', dest="no_unknown", action='store_true', default=False, help="Use if you'd like to add all subjects in the gallery. If this flag is set, \'unknown percentage\' will be ignored")
    arguments = parser.parse_args()

    # Checking arguments validity
    if not os.path.exists(arguments.dataset_path):
        print("Path folder not found.")
        exit(-1)

    if arguments.gallery_percentage < 0 or arguments.gallery_percentage >= 1:
        print("Argument size not valid. Please provide a decimal number between 0 and 0.9")
        exit(-1)

    if arguments.unknown_percentage < 0 or arguments.unknown_percentage >= 1:
        print("Argument size not valid. Please provide a decimal number between 0 and 0.9")
        exit(-1)


    gallery, probe = processDataset(arguments.dataset_path, arguments.gallery_percentage, arguments.unknown_percentage, arguments.no_unknown)
 
    for subject in gallery:
        for image in gallery[subject]:
            res = face_recognition.face_encodings(cv2.imread(f"{arguments.dataset_path}/{subject}/{image}"))
            if len(res) == 0: continue

            image_encoding = res[0]
            known_encodings.append(image_encoding)
            known_names.append(subject)

    # Calculating performance
    DIR_List = []
    FRR_List = []
    FAR_List = []
    GRR_List = []
    GAR_List = []
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for t in thresholds:
        DI = np.zeros(len(gallery)*len(gallery[0])); TG = len(probe); TI = len(probe)
        FA = 0; GR = 0 
        counter = 0
        
        for subject in probe:
            for image in probe[subject]:
                res = face_recognition.face_encodings(cv2.imread(f"{arguments.dataset_path}/{subject}/{image}"))
                if len(res) == 0: continue
                break

            if len(res) == 0: continue
            image_encoding = res[0]

            distances = face_recognition.face_distance(known_encodings, image_encoding)
            min_distance_index = distances.argmin()
            sorted_indices = np.argsort(distances)

            if distances[min_distance_index] < t:
                counter += 1
                name = known_names[min_distance_index]
                find = False
                if name == subject:
                    DI[0] += 1
                    for index in sorted_indices:
                        name_other = known_names[index]
                        if (name_other != subject and distances[index] < t):
                            FA += 1
                            find = True
                            break

                    if(find == False):
                        GR+=1
                else:
                    j = 0
                    for index in sorted_indices:
                        other_name = known_names[index]
                        if (other_name == subject and distances[i] < t and j != 0):
                            DI[i] += 1
                            find = True
                            break
                        j += 1
                    FA += 1
            else:
                GR += 1
                
        DIR = np.zeros(counter-1) 
        DIR[0] = DI[0]/TG 
        FRR = 1-DIR[0] 
        FAR = FA/TI 
        GRR = GR/TI 
        GAR = 1- FRR
    
        for i in range (1, counter-1):
            DIR[i] = DI[i]/(TG + DIR[i-1]) 

        DIR_List.append(DIR[0])
        FRR_List.append(FRR)
        FAR_List.append(FAR)
        GRR_List.append(GRR)
        GAR_List.append(GAR)
        
        print()
        print("------------------------------Start performance evaluation------------------------------")
        print("DIR-(Detect Identification Rate) is the probability that the subject is identified at rank i")
        print(f"\tProbability that the subject is identified at rank 0 is --> {DIR[0]}")
        print("")
        print("FRR-(False Rejection Rate) is the probability that a genuine subject is wrongly rejected")
        print(f"\tFRR is --> {FRR}")
        print("")
        print("FAR-(False Acceptance Rate) is the probability that an impostor is wrongly accepted")
        print(f"\tFAR is --> {FAR}")
        print("")
        print("GRR-(Good Rejection Rate) is the probability that an impostor is correctly rejected")
        print(f"\tGRR is --> {GRR}")
        print("GAR-(Genuine Acceptance Rate) is the probability that a genuine subject is correctly accepted")
        print(f"\tGAR is --> {GAR}")
        print("------------------------------Stop performance evaluation------------------------------")  

    print(f"DIR_List: {DIR_List}")
    print(f"FRR_List: {FRR_List}")
    print(f"FAR_List: {FAR_List}")
    print(f"GRR_List: {GRR_List}")
    print(f"GAR_List: {GAR_List}")
    
    plt.figure(1)
    plt.plot(thresholds, FRR_List)
    plt.title('FRR variation based on threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('FRR')
    
    plt.figure(2)
    plt.plot(thresholds, FAR_List)
    plt.title('FAR variation based on threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('FAR')
    
    plt.figure(3)
    plt.plot(thresholds, DIR_List)
    plt.title('DIR variation based on threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('DIR')
    
    plt.show()  
    
if __name__ == '__main__': 
    main()