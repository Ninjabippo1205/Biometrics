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
	test_subject = test_subject.split('/')[0]
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
	test_subjects = gallery_subjects[:25]
	test_subjects_2 = np.unique([elem.split('/')[0] for elem in d_keys[200:249]])#d_keys[200:249]	
	test_subjects = np.concatenate((test_subjects, test_subjects_2))
	test_subjects = np.unique(test_subjects)

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
		rand = np.random.randint(0, 1)
		L_or_R = ""
		if rand == 0:
			if os.path.exists(f"CASIA-Iris-Interval/{test_subject}/L"):
				L_or_R = "L"
			else:
				L_or_R = "R"
		else:
			if os.path.exists(f"CASIA-Iris-Interval/{test_subject}/R"):
				L_or_R = "R"
			else:
				L_or_R = "L"

		max = len(dataset[f"{test_subject}/{L_or_R}"])
		random = np.random.randint(0, max)
		probe_path = f"{arguments.path}/{f'{test_subject}/{L_or_R}'}/{dataset[f'{test_subject}/{L_or_R}'][random]}"
		if test_subject in gallery_subjects:
			probes_inside_gallery += 1
		else:
			probes_outside_gallery += 1
		probes.append([f"{test_subject}/{L_or_R}", probe_path])

	for threshold in range(250, 700, 25):#(2000, 20000 ,2000):
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

if __name__ == "__main__":
	main2()

#################################################### i test_subjects sono appartenenti sia alla gallery che non

###con immagine equalizzata e n_components=2 thresh(2000, 10000, 500)
#DIR_List:  ['0.0', '0.02', '0.02', '0.04', '0.06', '0.06', '0.06', '0.06', '0.1', '0.1', '0.12', '0.12', '0.14', '0.18', '0.2', '0.2']
#GRR_List:  ['1.0', '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '0.97', '0.79', '0.69','0.55','0.38', '0.28', '0.18', '0.16', '0.14','0.10']
#FRR_List:  ['1.0', '0.98', '0.98', '0.96', '0.94', '0.94', '0.94', '0.94', '0.9', '0.9', '0.88', '0.88', '0.86', '0.82', '0.8', '0.8']
#FAR_List:  ['0.0', '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.02', '0.20', '0.30','0.44','0.61', '0.71', '0.81', '0.83', '0.85','0.89']

###con immagine equalizzata e n_components=3 thresh(2000, 15000, 500)
#DIR_List:  ['0.0', '0.02', '0.02', '0.02', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.06', '0.08', '0.08', '0.1', '0.12', '0.12', '0.16', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18']
#GRR_List:  ['1.0', '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '0.95', '0.83', '0.77', '0.65', '0.55', '0.42','0.24', '0.14', '0.12', '0.12', '0.04', '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0']
#FRR_List:  ['1.0', '0.98', '0.98', '0.98', '0.96', '0.96', '0.96', '0.96', '0.96', '0.96', '0.96', '0.94', '0.92', '0.92', '0.9', '0.88', '0.88', '0.84', '0.82', '0.82', '0.82', '0.82', '0.82', '0.82', '0.82', '0.82']
#FAR_List:  ['0.0', '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.04', '0.16', '0.22', '0.34', '0.44', '0.57','0.75', '0.85', '0.87', '0.87', '0.95', '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0']

###con immagine equalizzata e n_components=4 thresh(2000, 15000, 500)
#DIR_List:  ['0.0', '0.0', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.04', '0.04', '0.04', '0.04', '0.08', '0.08', '0.08', '0.1', '0.12', '0.12', '0.12', '0.16', '0.16', '0.16', '0.16', '0.16', '0.16']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '0.95', '0.87', '0.75', '0.65', '0.55', '0.44','0.22', '0.14', '0.10', '0.06', '0.04', '0.0',  '0.0',  '0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.98', '0.98', '0.98', '0.98', '0.98', '0.98', '0.96', '0.96', '0.96', '0.96', '0.92', '0.92', '0.92', '0.9', '0.88', '0.88', '0.88', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.04', '0.12', '0.24', '0.34', '0.44', '0.55','0.77', '0.85', '0.89', '0.93', '0.95', '1.0',  '1.0',  '1.0',  '1.0']

###con immagine equalizzata e n_components=5 thresh(2000, 20000, 2000)
#DIR_List:  ['0.0', '0.02', '0.04', '0.04', '0.08', '0.08', '0.14', '0.14', '0.14']
#GRR_List:  ['1.0', '1.0',  '1.0',  '0.97', '0.53', '0.14', '0.04', '0.0',  '0.0']
#FRR_List:  ['1.0', '0.98', '0.96', '0.96', '0.92', '0.92', '0.86', '0.86', '0.86']
#FAR_List:  ['0.0', '0.0',  '0.0',  '0.02', '0.46', '0.85', '0.95', '1.0',  '1.0']

###con immagine equalizzata e n_components=7 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.02', '0.02', '0.04', '0.12', '0.14', '0.22', '0.22', '0.22']
#GRR_List:  ['1.0', '1.0',  '1.0',  '1.0',  '0.59', '0.18', '0.02', '0.0',  '0.0']
#FRR_List:  ['1.0', '0.98', '0.98', '0.96', '0.88', '0.86', '0.78', '0.78', '0.78']
#FAR_List:  ['0.0', '0.0',  '0.0',  '0.0',  '0.40', '0.81', '0.97', '1.0',  '1.0']

###con immagine equalizzata e n_components=8 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.04', '0.06', '0.1', '0.16', '0.16', '0.16']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.67', '0.30','0.06', '0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.96', '0.94', '0.9', '0.84', '0.84', '0.84']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.32', '0.69','0.93', '1.0',  '1.0']

###con immagine equalizzata e n_components=9 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.04', '0.04', '0.1', '0.16', '0.18', '0.18']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.81', '0.32','0.02', '0.02', '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.96', '0.96', '0.9', '0.84', '0.82', '0.82']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.18', '0.67','0.97', '0.97', '1.0']

###con immagine equalizzata e n_components=10 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.04', '0.04', '0.08', '0.1', '0.14', '0.14']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.81', '0.28', '0.10','0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.96', '0.96', '0.92', '0.9', '0.86', '0.86']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.18', '0.71', '0.89','1.0',  '1.0']

###con immagine equalizzata e n_components=11 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.02', '0.04', '0.06', '0.1', '0.14', '0.14']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.91', '0.32', '0.08','0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.98', '0.96', '0.94', '0.9', '0.86', '0.86']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.08', '0.67', '0.91','1.0',  '1.0']


###con immagine equalizzata e n_components=12 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.04', '0.04', '0.08', '0.1', '0.12', '0.12']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.91', '0.36', '0.04','0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.96', '0.96', '0.92', '0.9', '0.88', '0.88']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.08', '0.63', '0.95','1.0',  '1.0']

###con immagine equalizzata e n_components=13 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.02', '0.04', '0.06', '0.06', '0.08', '0.08']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '1.0',  '0.46', '0.14', '0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.98', '0.96', '0.94', '0.94', '0.92', '0.92']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.0',  '0.53', '0.85', '1.0',  '1.0']

###con immagine equalizzata e n_components=14 thresh(2000, 20000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.02', '0.04', '0.08', '0.1', '0.14', '0.14']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.95', '0.55', '0.14','0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.98', '0.96', '0.92', '0.9', '0.86', '0.86']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.04', '0.44', '0.85','1.0',  '1.0']


###con immagine equalizzata e n_components=15 thresh(2000, 40000 , 3000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.06', '0.06', '0.14', '0.14', '0.14', '0.14', '0.14', '0.14', '0.14', '0.14']
#GRR_List:  ['1.0', '1.0', '1.0',  '0.85', '0.08', '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.94', '0.94', '0.86', '0.86', '0.86', '0.86', '0.86', '0.86', '0.86', '0.86']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.14', '0.91', '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0',  '1.0']

###con immagine equalizzata e n_components=16 thresh(2000, 40000 , 3000)
#DIR_List:  ['0.0', '0.0', '0.0', '0.02', '0.04', '0.08', '0.1', '0.14', '0.14']
#GRR_List:  ['1.0', '1.0', '1.0', '1.0',  '0.95', '0.53', '0.10','0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '1.0', '0.98', '0.96', '0.92', '0.9', '0.86', '0.86']
#FAR_List:  ['0.0', '0.0', '0.0', '0.0',  '0.04', '0.46', '0.89','1.0',  '1.0']

## 17, 18, 19, 20, 21 25 sono una merda

###con immagine equalizzata e n_components=50 thresh(2000, 26000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.02', '0.04', '0.04', '0.08', '0.1', '0.12', '0.12', '0.12', '0.12']
#GRR_List:  ['1.0', '1.0', '1.0',  '1.0',  '0.85', '0.30', '0.04','0.0',  '0.0',  '0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '0.98', '0.96', '0.96', '0.92', '0.9', '0.88', '0.88', '0.88', '0.88']
#FAR_List:  ['0.0', '0.0', '0.0',  '0.0',  '0.14', '0.69', '0.95','1.0',  '1.0',  '1.0',  '1.0']

###con immagine equalizzata e n_components=100 thresh(2000, 26000 , 2000)
#DIR_List:  ['0.0', '0.0', '0.0', '0.02', '0.02', '0.06', '0.12', '0.12', '0.16', '0.16', '0.16', '0.16']
#GRR_List:  ['1.0', '1.0', '1.0', '1.0',  '1.0',  '0.85', '0.38', '0.10', '0.0',  '0.0',  '0.0',  '0.0']
#FRR_List:  ['1.0', '1.0', '1.0', '0.98', '0.98', '0.94', '0.88', '0.88', '0.84', '0.84', '0.84', '0.84']
#FAR_List:  ['0.0', '0.0', '0.0', '0.0',  '0.0',  '0.14', '0.61', '0.89', '1.0',  '1.0',  '1.0',  '1.0']

####################################################	tutti i test_subjects appartenenti alla gallery
#con immagine equalizzata e n_components=7 thresh(2000, 20000 , 2000)
