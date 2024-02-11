# project imports
import iris_identification
import iris_processing as IrisProcessing

# global imports
import cv2, os, multiprocessing, argparse
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

def dataset_chunks(dataset, parts):
	it = iter(dataset)
	size = int(len(dataset)/parts)
	for _ in range(0, len(dataset), size):
		yield {k:dataset[k] for k in islice(it, size)}

def viewImages(dataset, timeout):
	for i in dataset:
		for j in dataset[i]:
			for k in dataset[i][j]:
				frame = cv2.imread(k)
				cv2.imshow("input", frame)
				key = cv2.waitKey(timeout)
				if key == 27 or key == 1048603:
					break

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

def main():
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

	pool = multiprocessing.Pool(
		processes=arguments.processcount,
	)
	
	dataset = pool.apply(func=createDatasetfromPath, args=(arguments.path,))
	
	if arguments.template is True:
		dataset_arguments = []
		datasetchunks = dataset_chunks(dataset, arguments.parts)
		for _ in datasetchunks: dataset_arguments.append((_, arguments.path))
		pool.starmap(func=IrisProcessing.saveDataset, iterable=dataset_arguments)
		exit(0)

	d_keys = list(dataset.keys())

	gallery_subjects = np.unique([elem.split('/')[0] for elem in d_keys[:200]])
	test_subjects = gallery_subjects[:25]
	test_subjects_2 = np.unique([elem.split('/')[0] for elem in d_keys[200:249]])
	test_subjects = np.concatenate((test_subjects, test_subjects_2))
	test_subjects = np.unique(test_subjects)

	img = []
	for g_s in range(len(gallery_subjects)):
		if os.path.exists(f"{arguments.path}/{gallery_subjects[g_s]}/L"):
			img += os.listdir(f"{arguments.path}/{gallery_subjects[g_s]}/L")
		if os.path.exists(f"{arguments.path}/{gallery_subjects[g_s]}/R"):
			img += os.listdir(f"{arguments.path}/{gallery_subjects[g_s]}/R")

	DIR_List = []
	FRR_List = []
	FAR_List = []
	GRR_List = []

	thresholds = []
	probes = []

	probes_inside_gallery = 0
	probes_outside_gallery = 0

	for test_subject in test_subjects:
		choice  = np.random.choice([True, False]) # True is Left, False is Right

		# Choosing which eye
		if (choice and os.path.exists(f"{arguments.path}/{test_subject}/L")) or not os.path.exists(f"{arguments.path}/{test_subject}/R"): eye = "L"
		else: eye = "R"

		max = len(dataset[f"{test_subject}/{eye}"])
		random = np.random.randint(0, max)
		probe_path = f"{arguments.path}/{test_subject}/{eye}/{dataset[f'{test_subject}/{eye}'][random]}"

		if test_subject in gallery_subjects: probes_inside_gallery += 1
		else: probes_outside_gallery += 1
		probes.append([f"{test_subject}/{eye}", probe_path])

	for threshold in range(150, 650, 25):
		thresholds.append(threshold)

		print(f"-----------------------STARTING EVALUATION WITH THRESHOLD {threshold}-----------------------")
		FA = 0
		GR = 0
		FR = 0
		TG = probes_inside_gallery
		TI = probes_outside_gallery
		DI = np.zeros(len(gallery_subjects))

		for i in range(len(probes)):
			test_subject = probes[i][0]
			test_subject_path = probes[i][1]
			
			matched_list = iris_identification.image_matching(arguments.path, test_subject_path, gallery_subjects, threshold, pool)

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

	print("DIR_List: ", DIR_List)
	print("FRR_List: ", FRR_List)
	print("FAR_List: ", FAR_List)
	print("GRR_List: ", GRR_List)

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
	main()