import cv2, os, random, argparse, multiprocessing, face_recognition
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

def calculateImageEncoding(path, gallery, subject):
    items = []

    for image in gallery[subject]:
        res = face_recognition.face_encodings(cv2.imread(f"{path}/{subject}/{image}"))
        if len(res) == 0: continue

        items.append((res[0], subject))

    return items

def calculateParameters(path, gallery, probe, known_encodings, known_names, threshold):
    DI = np.zeros(len(gallery)*len(list(gallery.keys())[0])); TG = len(probe); TI = len(probe)
    FA = 0; GR = 0 
    counter = 0

    for subject in probe:
        for image in probe[subject]:
            res = face_recognition.face_encodings(cv2.imread(f"{path}/{subject}/{image}"))
            if len(res) == 0: continue
            break

        if len(res) == 0: continue
        image_encoding = res[0]
        distances = face_recognition.face_distance(known_encodings, image_encoding)
        min_distance_index = distances.argmin()
        sorted_indices = np.argsort(distances)
        if distances[min_distance_index] < threshold:
            counter += 1
            name = known_names[min_distance_index]
            find = False
            if name == subject:
                DI[0] += 1
                for index in sorted_indices:
                    name_other = known_names[index]
                    if (name_other != subject and distances[index] < threshold):
                        FA += 1
                        find = True
                        break
                if(find == False):
                    GR += 1
            else:
                j = 0
                for index in sorted_indices:
                    other_name = known_names[index]
                    if (other_name == subject and distances[index] < threshold and min_distance_index != index):
                        DI[j] += 1
                        break
                    j += 1
                FA += 1
        else:
            GR += 1

    DIR = np.zeros(counter-1) 
    DIR[0] = DI[0]/TG 
    FRR = 1 - DIR[0] 
    FAR = FA/TI 
    GRR = GR/TI 
    GAR = 1 - FRR

    for i in range (1, counter-1):
        DIR[i] = DI[i]/(TG + DIR[i-1])
    return [DIR[0], FRR, FAR, GRR, GAR]

def main():
    # Moving inside folder (os dependent)
    current_folder = os.path.realpath(__file__)
    if os.name == 'nt': os.chdir(current_folder[:current_folder.rfind('\\')])
    elif os.name == 'posix': os.chdir(current_folder[:current_folder.rfind('/')])
    else:
        print("This program hasn't been tested in the current enviroment. Downtime may happen.")
        _ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
        if _ != '': exit(-1)
        os.chdir(current_folder[:current_folder.rfind('/')])

    known_encodings = []
    known_names = []

    parser = argparse.ArgumentParser(
		prog="FaceRecognition",
		description="This program recognizes faces based on a face recognition dataset",
		epilog="The dataset must be created as such: \"dataset/xxx/xxx_y.bmp\" where \'x\' is the subject's number and \'y\' the numbered photo."
	)
    
    parser.add_argument('-p', '--path', dest="dataset_path", type=str, default="CASIA-FaceV5", help="Path to the folder where the face images are saved. Provide the path with the absolute rapresentation")
    parser.add_argument('-gp', '--gallery-percentage', dest="gallery_percentage", default="0.8", type=float, help="Percentage of images for chosen subjects that will be added to the gallery")
    parser.add_argument('-up', '--unkown-percentage', dest="unknown_percentage", default="0.2", type=float, help="Percentage of unkown subjects that are going to be tested" )
    parser.add_argument('-nu', '--no-unknown', dest="no_unknown", action='store_true', default=False, help="Use if you'd like to add all subjects in the gallery. If this flag is set, \'unknown percentage\' will be ignored")
    parser.add_argument('-pc', '--process-count', dest="processcount", type=int,
									 default=int(multiprocessing.cpu_count()*3/4), help="Amount of processes avbailable for this program's execution")
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
    
    if arguments.processcount < 0 or arguments.processcount == 0:
        print("Given process argument is invalid.")
        exit(-1)
    elif arguments.processcount > multiprocessing.cpu_count():
        print("The number of process given is more than the number of cpu cores. This may cause the system to slow down.")
        _ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
        if _ != '': exit(-1)

    
    process_pool = multiprocessing.Pool(
        processes = arguments.processcount,
        maxtasksperchild = 2
    )

    print("----------------------------------- Generating dataset -----------------------------------")
    gallery, probe = processDataset(arguments.dataset_path, arguments.gallery_percentage, arguments.unknown_percentage, arguments.no_unknown)

    args = []
    for subject in gallery:
        args.append([arguments.dataset_path, gallery, subject]) 
    
    print("--------------------------------- Started Image encoding ---------------------------------")
    res = process_pool.starmap(func=calculateImageEncoding, iterable=args)
    print("-------------------------------- Finished Image encoding ---------------------------------")
    for process_result in res:
        for item in process_result:
            known_encodings.append(item[0])
            known_names.append(item[1])
    args.clear()
    res.clear()

    print("----------------------------- Started performance evaluation -----------------------------")
    print("DIR - (Detect Identification Rate) - is the probability that the subject is identified at rank i")
    print("FRR - (False Rejection Rate) - is the probability that a genuine subject is wrongly rejected")
    print("FAR - (False Acceptance Rate) - is the probability that an impostor is wrongly accepted")
    print("GRR - (Good Rejection Rate) - is the probability that an impostor is correctly rejected")
    print("GAR - (Genuine Acceptance Rate) - is the probability that a genuine subject is correctly accepted")

    # Calculating performance
    DIR_List = []
    FRR_List = []
    FAR_List = []
    GRR_List = []
    GAR_List = []
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    args = []
    for t in thresholds:
        args.append([arguments.dataset_path, gallery, probe, known_encodings, known_names, t])

    items = process_pool.starmap(func=calculateParameters, iterable=args)
  
    process_pool.close()
    process_pool.join()
    args.clear()

    for item in items:
        DIR_List.append(item[0])
        FRR_List.append(item[1])
        FAR_List.append(item[2])
        GRR_List.append(item[3])
        GAR_List.append(item[4])
        

    print(f"DIR_List: {DIR_List}")
    print(f"FRR_List: {FRR_List}")
    print(f"FAR_List: {FAR_List}")
    print(f"GRR_List: {GRR_List}")
    print(f"GAR_List: {GAR_List}")
    print("------------------------------ Performance evaluation finished ------------------------------")
    
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