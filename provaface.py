import face_recognition 
import cv2 
import os 
import numpy as np 
import random 
import face_recognition 
import matplotlib.pyplot as plt
 
def processDataset(path, gallery_percentage, unkown_percentage): 
    if not os.path.exists(path): 
        print("Dataset could not be loaded, please check folder location") 
        exit(-1) 
 
    gallery = {} 
    probe = {} 
 
    subjects = os.listdir(path) 
     
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
    gallery, probe = processDataset('CASIA-FaceV5', 0.80, 0.2 )
 
    for key in gallery: 
        list_img=gallery[key] 
        for img in list_img: 
            tmp = face_recognition.face_encodings(cv2.imread(f"CASIA-FaceV5/{key}/{img}")) 
            if len(tmp)==0: continue
            
            img_enc = tmp[0]
            known_encodings.append(img_enc) 
            known_names.append(key)
                

    print("probe")
    DIR_List = []
    FRR_List = []
    FAR_List = []
    GRR_List = []
    GAR_List = []
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for t in thresholds:
        DI=np.zeros(len(gallery)*5); TG=len(probe); TI=len(probe)
        FA=0; GR=0;  
        counter = 0
        
        for key in probe: 
            list_img=probe[key]
            label= key 

            for i in list_img:
                tmp = face_recognition.face_encodings(cv2.imread(f"CASIA-FaceV5/{key}/{i}"))
                if len(tmp)==0: continue
                break
            
            if len(tmp)==0: continue
            img_enc = tmp[0]
    
            distances = face_recognition.face_distance(known_encodings, img_enc) 
            min_distance_index = distances.argmin() 
            sorted_indices = np.argsort(distances) 
            #print(label, known_names[min_distance_index], distances[min_distance_index])

            if distances[min_distance_index] < t:  
                counter += 1
                name = known_names[min_distance_index]              
                find=False 
                if name==label:  
                    DI[0]+=1 
                    for i in sorted_indices: 
                        name_other = known_names[i] 
                        if (name_other!=label and distances[i]< t): 
                            FA+=1 
                            find=True 
                            break 
                    
                    if(find==False): 
                        GR+=1 
                else: 
                    j=0 
                    for i in sorted_indices: 
                        name_other = known_names[i] 
                        if (name_other==label and distances[i]< t and j!=0): 
                            DI[i]+=1 
                            find=True 
                            break 
                        j+=1 
                    FA+=1 
            else: 
                GR+=1 
                
        DIR=np.zeros(counter-1) 
        DIR[0]=DI[0]/TG 
        FRR=1-DIR[0] 
        FAR=FA/TI 
        GRR=GR/TI 
        GAR = 1- FRR
    
        for i in range (1, counter-1): 
                DIR[i]= DI[i]/(TG+ DIR[i-1]) 

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

    print("DIR_List: ",DIR_List)
    print("FRR_List: ",FRR_List)
    print("FAR_List: ",FAR_List)
    print("GRR_List: ",GRR_List)
    print("GAR_List: ",GAR_List)
    
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