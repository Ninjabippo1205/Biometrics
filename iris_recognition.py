# project imports
import iris_identification
import iris_processing as IrisProcessing

# global imports
import cv2, os, random, multiprocessing, argparse
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt

def createDatasetfromPath(path):
	dataset = {}
	for folder in os.listdir(path):
		# Adding left eye images
		if os.path.exists(f"CASIA-Iris-Lamp/{folder}/L"): # (as long as it exists)
			dataset[f"{folder}/L"] = os.listdir("CASIA-Iris-Lamp/" + folder + "/" + "L")

		# Adding right eye images
		if os.path.exists(f"CASIA-Iris-Lamp/{folder}/R"):
			dataset[f"{folder}/R"] = os.listdir("CASIA-Iris-Lamp/" + folder + "/" + "R")
	return dataset

def viewImages(dataset, timeout):
	for i in dataset:
		for j in dataset[i]:
			for k in dataset[i][j]:
				frame = cv2.imread(k)
				cv2.imshow("input", frame)
				key = cv2.waitKey(timeout)
				if key == 27 or key == 1048603:
					break


def dataset_chunks(dataset, parts):
	it = iter(dataset)
	size = int(len(dataset)/parts)
	for _ in range(0, len(dataset), size):
		yield {k:dataset[k] for k in islice(it, size)}


def main():
	parser = argparse.ArgumentParser(
		prog="IrisRecognition",
		description="This program handles enrlomment and recognition of someone's iris",
		epilog=""
	)

	parser.add_argument('-p', '--path', dest="path", type=str, default="CASIA-Iris-Lamp", help="Folder where iris images are saved")
	parser.add_argument('-pc', '--process-count', dest="processcount", type=int,
										 default=int(multiprocessing.cpu_count()*3/4), help="Amount of processes allocated for this program")
	
	parser.add_argument('-t', '--template', dest='template', action='store_true',
										  default=False, help="To be used if you want to calculate every image template")
	parser.add_argument('--parts', dest='parts', default=50, type=int,
										 help="Amount of parts to divide the dataset; used for template creation. WARNING! The -t flag must be set")
	
	arguments = parser.parse_args()
	if not os.path.exists(arguments.path):
		print("Path folder not found.")
		exit(-1)
	
	if arguments.processcount < 0 or arguments.processcount == 0:
		print("Given process argument is invalid.")
		exit(-1)
	elif arguments.processcount > multiprocessing.cpu_count():
		print("The number of process given is more than the number of cpu cores. This may cause the system to slow down.")
		_ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
		if _ != '' or _ == 'q':
			exit(-1)

	pool = multiprocessing.Pool(
    processes=arguments.processcount,
    #maxtasksperchild = 1 # Keep commented for maximum performace
  )
	
	# Create dataset
	dataset = pool.apply(func=createDatasetfromPath, args=(arguments.path,))
	
	# Checking if the entire dataset template calulation is asked
	if arguments.template is True:
		# Creating dataset arguments to divide dataset evenly. This will distribute the images creations to all processes
		dataset_arguments = []
		datasetchunks = dataset_chunks(dataset, arguments.parts)
		for _ in datasetchunks: dataset_arguments.append((_, arguments.path))

		pool.starmap(func=IrisProcessing.saveDataset, iterable=dataset_arguments)
		exit(0)

	# Saving all templates using allocated processes.
	pool.apply_async(func=IrisProcessing.saveDataset, args=(dataset, arguments.path,))

	d_keys = list(dataset.keys());# random.shuffle(d_keys)		#all keys of dataset
	
	# Get a test subject based on the shuffled keys
	#test_subject = d_keys[random.randint(0, len(d_keys)-1)]
	#probe = dataset[test_subject] # Probe is a list of images

	# Using the first 20 elements to use as gallery
	gallery_subjects = d_keys[:200]							#keys in the gallery, the template is calculated only if not alredy stored
	#random.shuffle(d_keys)
	test_set = d_keys[:25]				#keys to be tested, the template will be calculated anyway
	test_set = test_set + d_keys[300:325]
	# Checking that there is both left and right eye for every subject
	for eye in gallery_subjects:
		# 158 is ascii for L+R. By removing a letter, the other ascii number will pop up
		other_eye = eye[:-1]+chr(158 - ord(eye[-1]))
		if not other_eye in gallery_subjects and os.path.exists(f'{arguments.path}/{other_eye}'): gallery_subjects.append(other_eye)


	gallery = {} # Gallery is a subset of the dictionary "dataset"
	for x in gallery_subjects: gallery[x] = dataset[x]

	DIR_List = []
	FRR_List = []
	FAR_List = []
	GRR_List = []
	GAR_List = []
	thresholds = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

	for threshold in thresholds:
		print(f"-----------------------STARTING EVALUATION WITH THRESHOLD {threshold}-----------------------")
		# Calculating False Acceptance, Good Rejection, Detect Indentification, Total Genuine and Total Impostor
		#									(yes|no),				(no|no),						(yes|yes)
		FA = 0; GR = 0; TG =len(gallery_subjects); TI = len(d_keys)
		DI = np.zeros(len(gallery_subjects*40)-1)

		## image matching ##
		#for test_subject in d_keys:
		for test_subject in test_set:
			probe = dataset[test_subject][0]
		
			find = False
			subject_matched, minimum_distance, matched_list = iris_identification.image_matching(arguments.path, test_subject, probe, gallery, gallery_subjects, threshold, pool)

			if(minimum_distance < threshold):
				if(subject_matched == test_subject):
					DI[0] = DI[0] + 1
					for m in matched_list.keys():
						if m != test_subject:
							FA += 1
							find = True
							break

					if find == False:
						GR += 1	
				else:
					i = 0
					for m in matched_list.keys():
						if m == test_subject and i != 0:
							DI[i] += 1
							break

						i += 1
					FA += 1
					

				print(f"The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects}")
				os.system(f"echo The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects} >> result.txt")
			else:
				GR = GR + 1
				print(f"Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects}")
				os.system(f"echo Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects} >> result.txt")

		#pool.close()
			
		DIR = np.zeros(len(gallery_subjects)-1)
		DIR[0]= DI[0]/TG
		FRR = 1 - DIR[0]
		FAR = FA / TI
		GRR = 1 - FAR
		GAR = 1- FRR

		for i in range (1, len(gallery_subjects)-1):
			DIR[i]= DI[i]/(TG+ DIR[i-1])

		DIR_List.append(DIR[0])
		FRR_List.append(FRR)
		FAR_List.append(FAR)
		GRR_List.append(GRR)
		GAR_List.append(GAR)


		print()
		print("------------------------------Start performance evaluation------------------------------")
		print("DIR-(Detect Identification Rate) is the probability that the subject is identified at rank i")
		k = 10 #len(DIR)
		for i in range(k):
			print(f"\tProbability that the subject is identified at rank {i+1} is --> {DIR[i]}")
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
	pool.close()
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

if __name__ == "__main__":
	main()