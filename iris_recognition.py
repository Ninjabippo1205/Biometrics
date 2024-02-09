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
		if os.path.exists(f"CASIA-Iris-Interval/{folder}/L"): # (as long as it exists)
			dataset[f"{folder}/L"] = os.listdir("CASIA-Iris-Interval/" + folder + "/" + "L")

		# Adding right eye images
		if os.path.exists(f"CASIA-Iris-Interval/{folder}/R"):
			dataset[f"{folder}/R"] = os.listdir("CASIA-Iris-Interval/" + folder + "/" + "R")
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


#def main():
#	parser = argparse.ArgumentParser(
#		prog="IrisRecognition",
#		description="This program handles enrlomment and recognition of someone's iris",
#		epilog=""
#	)
#
#	parser.add_argument('-p', '--path', dest="path", type=str, default="CASIA-Iris-Interval", help="Folder where iris images are saved")
#	parser.add_argument('-pc', '--process-count', dest="processcount", type=int,
#										 default=int(multiprocessing.cpu_count()*3/4), help="Amount of processes allocated for this program")
#	
#	parser.add_argument('-t', '--template', dest='template', action='store_true',
#										  default=False, help="To be used if you want to calculate every image template")
#	parser.add_argument('--parts', dest='parts', default=50, type=int,
#										 help="Amount of parts to divide the dataset; used for template creation. WARNING! The -t flag must be set")
#	
#	arguments = parser.parse_args()
#	if not os.path.exists(arguments.path):
#		print("Path folder not found.")
#		exit(-1)
#	
#	if arguments.processcount < 0 or arguments.processcount == 0:
#		print("Given process argument is invalid.")
#		exit(-1)
#	elif arguments.processcount > multiprocessing.cpu_count():
#		print("The number of process given is more than the number of cpu cores. This may cause the system to slow down.")
#		_ = input('Please hit enter if you\'d like to continue, \'q\' otherwise: ')
#		if _ != '' or _ == 'q':
#			exit(-1)
#
#	pool = multiprocessing.Pool(
#    processes=arguments.processcount,
#    #maxtasksperchild = 1 # Keep commented for maximum performace
#  )
#	
#	# Create dataset
#	dataset = pool.apply(func=createDatasetfromPath, args=(arguments.path,))
#	
#	# Checking if the entire dataset template calulation is asked
#	if arguments.template is True:
#		# Creating dataset arguments to divide dataset evenly. This will distribute the images creations to all processes
#		dataset_arguments = []
#		datasetchunks = dataset_chunks(dataset, arguments.parts)
#		for _ in datasetchunks: dataset_arguments.append((_, arguments.path))
#
#		pool.starmap(func=IrisProcessing.saveDataset, iterable=dataset_arguments)
#		exit(0)
#
#	# Saving all templates using allocated processes.
#	pool.apply_async(func=IrisProcessing.saveDataset, args=(dataset, arguments.path,))
#
#	d_keys = list(dataset.keys())
#	# random.shuffle(d_keys)
#	
#	# Get a test subject based on the shuffled keys
#	#test_subject = d_keys[random.randint(0, len(d_keys)-1)]
#	#probe = dataset[test_subject] # Probe is a list of images
#
#	# Using the first 20 elements to use as gallery
#	gallery_subjects = d_keys[:100]		#max 395						#keys in the gallery, the template is calculated only if not alredy stored
#	#random.shuffle(d_keys)
#	test_set = d_keys[:20]									#keys to be tested, the template will be calculated anyway
#	#random.shuffle(d_keys)
#	#test_set = test_set + d_keys[300:310]
#	# Checking that there is both left and right eye for every subject
#	for eye in gallery_subjects:
#		# 158 is ascii for L+R. By removing a letter, the other ascii number will pop up
#		other_eye = eye[:-1]+chr(158 - ord(eye[-1]))
#		if not other_eye in gallery_subjects and os.path.exists(f'{arguments.path}/{other_eye}'): gallery_subjects.append(other_eye)
#
#
#	gallery = {} # Gallery is a subset of the dictionary "dataset"
#	for x in gallery_subjects: gallery[x] = dataset[x]
#
#	DIR_List = []
#	FRR_List = []
#	FAR_List = []
#	GRR_List = []
#	GAR_List = []
#
#	thresholds = []
#	for threshold in range(3500, 9000 ,100):
#		thresholds.append(threshold)
#		print(f"-----------------------STARTING EVALUATION WITH THRESHOLD {threshold}-----------------------")
#		# Calculating False Acceptance, Good Rejection, Detect Indentification, Total Genuine and Total Impostor
#		#									(yes|no),				(no|no),						(yes|yes)
#		FA = 0; GR = 0; TG =len(test_set); TI = len(test_set)
#		DI = np.zeros(len(gallery_subjects*40)-1)
#
#		## image matching ##
#		#for test_subject in d_keys:
#		for test_subject in test_set:
#			probe = dataset[test_subject][0]
#		
#			find = False
#			subject_matched, minimum_distance, matched_list = iris_identification.image_matching(arguments.path, test_subject, probe, gallery, gallery_subjects, threshold, pool)
#
#			if(minimum_distance < threshold):
#				if subject_matched == test_subject or subject_matched == test_subject[:-1]+chr(158 - ord(test_subject[-1])):
#					DI[0] = DI[0] + 1
#					for m in matched_list.keys():
#						if m != test_subject and m != test_subject[:-1]+chr(158 - ord(test_subject[-1])):
#							FA += 1
#							find = True
#							break
#
#					if find == False:
#						GR += 1	
#				else:
#					i = 0
#					for m in matched_list.keys():
#						if m == test_subject or m == test_subject[:-1]+chr(158 - ord(test_subject[-1])) and i != 0:
#							DI[i] += 1
#							break
#
#						i += 1
#					FA += 1
#					
#
#				print(f"The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects}")
#				os.system(f"echo The function was given {test_subject} to test. It has matched {subject_matched} with minimum distance {minimum_distance}. The gallery contained {test_subject}? {test_subject in gallery_subjects} >> result.txt")
#			else:
#				GR = GR + 1
#				print(f"Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects}")
#				os.system(f"echo Test subject not found! The function was given {test_subject}. The minimum distance found is: {minimum_distance} with {subject_matched}. Does the gallery contain the subject? {test_subject in gallery_subjects} >> result.txt")
#
#		#pool.close()
#			
#		DIR = np.zeros(len(gallery_subjects)-1)
#		DIR[0]= DI[0]/TG
#		FRR = 1 - DIR[0]
#		FAR = FA / TI
#		GRR = 1 - FAR
#		GAR = 1- FRR
#
#		for i in range (1, len(gallery_subjects)-1):
#			DIR[i]= DI[i]/(TG+ DIR[i-1])
#
#		DIR_List.append(DIR[0])
#		FRR_List.append(FRR)
#		FAR_List.append(FAR)
#		GRR_List.append(GRR)
#		GAR_List.append(GAR)
#
#
#		print()
#		print("------------------------------Start performance evaluation------------------------------")
#		print("DIR-(Detect Identification Rate) is the probability that the subject is identified at rank i")
#		k = 10 #len(DIR)
#		for i in range(k):
#			print(f"\tProbability that the subject is identified at rank {i+1} is --> {DIR[i]}")
#		print("")
#		print("FRR-(False Rejection Rate) is the probability that a genuine subject is wrongly rejected")
#		print(f"\tFRR is --> {FRR}")
#		print("")
#		print("FAR-(False Acceptance Rate) is the probability that an impostor is wrongly accepted")
#		print(f"\tFAR is --> {FAR}")
#		print("")
#		print("GRR-(Good Rejection Rate) is the probability that an impostor is correctly rejected")
#		print(f"\tGRR is --> {GRR}")
#		print("GAR-(Genuine Acceptance Rate) is the probability that a genuine subject is correctly accepted")
#		print(f"\tGAR is --> {GAR}")
#		print("------------------------------Stop performance evaluation------------------------------")
#
#	pool.close()
#	print("DIR_List: ",DIR_List)
#	print("FRR_List: ",FRR_List)
#	print("FAR_List: ",FAR_List)
#	print("GRR_List: ",GRR_List)
#	print("GAR_List: ",GAR_List)
#
#	plt.figure(1)
#	plt.plot(thresholds, FRR_List)
#	plt.title('FRR variation based on threshold values')
#	plt.xlabel('Threshold')
#	plt.ylabel('FRR')
#
#	plt.figure(2)
#	plt.plot(thresholds, FAR_List)
#	plt.title('FAR variation based on threshold values')
#	plt.xlabel('Threshold')
#	plt.ylabel('FAR')
#
#	plt.figure(3)
#	plt.plot(thresholds, DIR_List)
#	plt.title('DIR variation based on threshold values')
#	plt.xlabel('Threshold')
#	plt.ylabel('DIR')
#
#	plt.show()
#	return 0

def main2():
	dataset = createDatasetfromPath("CASIA-Iris-Interval")
	d_keys = list(dataset.keys())
	#random.shuffle(d_keys)	
	gallery_subjects = d_keys[:60]
	test_subjects = d_keys[:40]
	# random.shuffle(d_keys)

	for eye in gallery_subjects:
		# 158 is ascii for L+R. By removing a letter, the other ascii number will pop up
		other_eye = eye[:-1]+chr(158 - ord(eye[-1]))
		if not other_eye in gallery_subjects and os.path.exists(f'"CASIA-Iris-Interval/"{other_eye}'): gallery_subjects.append(other_eye)

	gallery = {} # Gallery is a subset of the dictionary "dataset"
	for x in gallery_subjects: gallery[x] = dataset[x]

	## image matching ##
	#for test_subject in d_keys:
	threshold = 8000
	for test_subject in test_subjects:
		print(test_subject)
		for img in dataset[test_subject]:
			#print(f"CASIA-Iris-Interval/{test_subject}/{img}")
			res = IrisProcessing.getTemplate(cv2.imread(f"CASIA-Iris-Interval/{test_subject}/{img}"))
		
	return 0

#def main3():
#	# Itera su tutte le sottocartelle da 1 a 249
#	users = os.listdir("CASIA-Iris-Interval")
#	for user in users:
#		eyes = os.listdir(f"CASIA-Iris-Interval/{user}")
#		for eye in eyes:
#			images = os.listdir(f"CASIA-Iris-Interval/{user}/{eye}")
#			if len(images) == 0:
#				os.rmdir(f"CASIA-Iris-Interval/{user}/{eye}")
#				print(f"rimossa cartella {eye} dell'utente {user}")

def metrics_calculator(matched_list, test_subject, probe, gallery_subjects):
	DI = np.zeros(len(gallery_subjects))
	FA = 0
	GR = 0
	FR = 0

	if len(matched_list) > 0:
		if test_subject not in gallery_subjects:
			FA = 1											#FA = 1 if the test_subject is not in the gallery but he found at least one match
		else:
			for i in range(len(matched_list)):
				if matched_list[i][0] == test_subject:		#if the test_subject is in the gallery and he found a match
					DI[i] = 1
			if matched_list[0][0] != test_subject:
				FR = 1										#FA = 1 if the test_subject is in the gallery but he didn't find a match
				
		print(f"The function was given {test_subject} to test. It has matched {matched_list[0][0]} with minimum distance {matched_list[0][1]}. The gallery contained {test_subject}? {test_subject in gallery_subjects}")
		os.system(f"echo The function was given {test_subject} to test. It has matched {matched_list[0][0]} with minimum distance {matched_list[0][1]}. The gallery contained {test_subject}? {test_subject in gallery_subjects} >> result.txt")
	else:
		if test_subject in gallery_subjects:
			FR = 1
		else:
			GR = 1
		print(f"Test subject not found! The function was given {test_subject}. Does the gallery contain the subject? {test_subject in gallery_subjects}")
		os.system(f"echo Test subject not found! The function was given {test_subject}. Does the gallery contain the subject? {test_subject in gallery_subjects} >> result.txt")
	return [FA, DI, GR, FR]

def main4():
	parser = argparse.ArgumentParser(
		prog="IrisRecognition",
		description="This program handles enrlomment and recognition of someone's iris",
		epilog=""
	)

	parser.add_argument('-p', '--path', dest="path", type=str, default="CASIA-Iris-Interval", help="Folder where iris images are saved")
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

	pool = multiprocessing.Pool(processes=
							 arguments.processcount,
							 )
	
	dataset = pool.apply(func=createDatasetfromPath, args=(arguments.path,))
	
	if arguments.template is True:
		dataset_arguments = []
		datasetchunks = dataset_chunks(dataset, arguments.parts)
		for _ in datasetchunks: dataset_arguments.append((_, arguments.path))
		pool.starmap(func=IrisProcessing.saveDataset, iterable=dataset_arguments)
		exit(0)

	#pool.apply_async(func=IrisProcessing.saveDataset, args=(dataset, arguments.path,))

	d_keys = list(dataset.keys())
	# random.shuffle(d_keys)
	
	#random.shuffle(d_keys)
	gallery_subjects = np.unique([elem.split('/')[0] for elem in d_keys[:200]])	#max 395
	#random.shuffle(d_keys)
	test_subjects = d_keys[:50]
	test_subjects = test_subjects + d_keys[200:249]	

	img = []
	for g_s in range(len(gallery_subjects)):
		if os.path.exists(f"CASIA-Iris-Interval/{gallery_subjects[g_s]}/L"):
			img += os.listdir(f"CASIA-Iris-Interval/{gallery_subjects[g_s]}/L")
		if os.path.exists(f"CASIA-Iris-Interval/{gallery_subjects[g_s]}/R"):
			img += os.listdir(f"CASIA-Iris-Interval/{gallery_subjects[g_s]}/R")

	DIR_List = []
	FRR_List = []
	FAR_List = []
	GRR_List = []

	thresholds = []
	probes = []
	probes_inside_gallery = 0
	probes_outside_gallery = 0

	for test_subject in test_subjects:
		max = len(dataset[test_subject])
		random = np.random.randint(0, max)
		probe_path = f"{arguments.path}/{test_subject}/{dataset[test_subject][random]}"
		if test_subject[:-2] in gallery_subjects:
			probes_inside_gallery += 1
		else:
			probes_outside_gallery += 1
		probes.append([test_subject[:-2], probe_path])

	for threshold in range(2000, 10000 ,500):
		thresholds.append(threshold)

		print(f"-----------------------STARTING EVALUATION WITH THRESHOLD {threshold}-----------------------")
		FA = 0
		GR = 0
		FR = 0
		TG =probes_inside_gallery
		TI = probes_outside_gallery
		DI = np.zeros(len(gallery_subjects))

		for i in range(len(probes)):
			test_subject = probes[i][0]
			test_subject_path = probes[i][1]
			
			matched_list = iris_identification.image_matching(arguments.path, test_subject, test_subject_path, gallery_subjects, threshold, pool)

			metrics = metrics_calculator(matched_list, test_subject, test_subject_path, gallery_subjects)
			FA += metrics[0]
			DI += metrics[1]
			GR += metrics[2]
			FR += metrics[3]
			
		DIR = np.zeros(len(gallery_subjects))
		DIR[0]= DI[0]/TG
		FRR = 1 - DIR[0]
		FAR = FA / TI
		GRR = 1 - FAR

		for i in range (1, len(gallery_subjects)-1):
			DIR[i]= DI[i]/(TG + DIR[i-1])

		DIR_List.append(str(DIR[0])[:4])
		FRR_List.append(str(FRR)[:4])
		FAR_List.append(str(FAR)[:4])
		GRR_List.append(str(GRR)[:4])

		print()
		print("------------------------------Start performance evaluation------------------------------")
		print("Number of probes inside the gallery: ", probes_inside_gallery)
		print("Number of probes outside the gallery: ", probes_outside_gallery)
		print("DIR-(Detect Identification Rate) is the probability that the subject is identified at rank i")

		k = 10
		if len(DIR) < 10:
			k = len(DIR)-1	

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
		print("------------------------------Stop performance evaluation------------------------------")

	pool.close()

	print("DIR_List: ",DIR_List)
	print("FRR_List: ",FRR_List)
	print("FAR_List: ",FAR_List)
	print("GRR_List: ",GRR_List)

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

	#plt.figure(3)
	#plt.plot(thresholds, DIR_List)
	#plt.title('DIR variation based on threshold values')
	#plt.xlabel('Threshold')
	#plt.ylabel('DIR')

	plt.figure(3)
	plt.plot(FAR_List, DIR_List)
	plt.title('ROC: DIR variation based on FAR values')
	plt.xlabel('FAR')
	plt.ylabel('DIR')

	plt.figure(4)
	plt.plot(FAR_List, FRR_List)
	plt.title('DET: FRR variation based on FAR values')
	plt.xlabel('FAR')
	plt.ylabel('FRR')

	plt.show()

	return 0

###con immagine equalizzata e n_components=2
#DIR_List:  [0.0, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.06, 0.08, 0.08, 0.08, 0.1, 0.14, 0.14, 0.16, 0.18]
#FRR_List:  [1.0, 0.98, 0.98, 0.98, 0.98, 0.96, 0.96, 0.94, 0.92, 0.92, 0.92, 0.9, 0.86, 0.86, 0.84, 0.8200000000000001]
#FAR_List:  [0.0, 0.0, 0.0, 0.0, 0.02040816326530612, 0.02040816326530612, 0.10204081632653061, 0.22448979591836735, 0.3469387755102041, 0.46938775510204084, 0.5714285714285714, 0.673469387755102, 0.7551020408163265, 0.8571428571428571, 0.8979591836734694, 0.9387755102040817]
#GRR_List:  [1.0, 1.0, 1.0, 1.0, 0.9795918367346939, 0.9795918367346939, 0.8979591836734694, 0.7755102040816326, 0.653061224489796, 0.5306122448979591, 0.4285714285714286, 0.326530612244898, 0.24489795918367352, 0.1428571428571429, 0.10204081632653061, 0.061224489795918324]

###con immagine originale e n_components=2
#DIR_List:  ['0.04', '0.04', '0.06', '0.1', '0.1', '0.12', '0.12', '0.18', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']
#FRR_List:  ['0.96', '0.96', '0.94', '0.9', '0.9', '0.88', '0.88', '0.82', '0.8', '0.8', '0.8', '0.8', '0.8', '0.8', '0.8', '0.8']
#FAR_List:  ['0.0', '0.0', '0.12', '0.22', '0.42', '0.73', '0.77', '0.87', '0.93', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0']
#GRR_List:  ['1.0', '1.0', '0.87', '0.77', '0.57', '0.26', '0.22', '0.12', '0.06', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']

if __name__ == "__main__":
	main4()