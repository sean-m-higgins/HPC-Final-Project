from knn import Knn, KnnParallel
import numpy as np
import datetime
import math
import multiprocessing
from multiprocessing import Process, Pool
from sklearn.metrics import accuracy_score


def run_knn(n, p, k, parallel, num_procs):
	#initialize random data here
	if n > 0 and p > 0:
		np.random.seed(42)
		X_train = np.random.rand(int(n*.8), p)
		y_train = np.concatenate((np.zeros(int((n/2)*.8)), np.ones(int((n/2)*.8))))
		X_dev = np.random.rand(int(n*.2), p)
		y_dev = np.concatenate((np.zeros(int((n/2)*.2)), np.ones(int((n/2)*.2))))
			
	one = str(datetime.datetime.now().time()).split(":")

	if parallel:
		knn = KnnParallel(k, X_dev, X_train, y_train, num_procs)
	else:
		knn = Knn(k, X_dev, X_train, y_train)

	two = str(datetime.datetime.now().time()).split(":")

	predictions  = knn.predictions
	accuracy = accuracy_score(y_dev, predictions)

	len_X_train = len(X_train)
	# KNN cost = run() --> get_neighbors() + get_majority_vote()
	# = (for each X_dev example) * { [get its neighbors] + [then get the top vote] }
	# = (# of X_dev examples) * ...
	# ...{ [ (# of X_train * ) + (sort # of X_train) + (k)] + [ (k*2) + (k) ] }
	cost = len(X_dev) * ( ( (len_X_train*(2 + 4 + p*2)) + 
		(len_X_train * math.log(len_X_train)) + (k) ) + (k*2 + k) )
	time_diff = (float(two[0]) - float(one[0]))*3600 + 
		(float(two[1]) - float(one[1]))*60 + (float(two[2]) - float(one[2]))

	print(str(n) + ", " + str(p) + ", " + str(k) + ", " + str(int(cost)) + ", " 
		str(time_diff) + ", " + str(parallel) + ", " + str(num_procs) + ", " + str(accuracy))


if __name__ == "__main__":

	# run_set_n = int(input("Enter Initial # of rows: "))
	# run_set_p = int(input("Enter Initial # of colums: "))
	# run_set_k = int(input("Enter Initial k neighbors: "))
	
	# get the number of CPUs 
	# num_procs = multiprocessing.cpu_count()
	# print('You have {0:1d} CPUs'.format(num_procs))

	# run_set = []
	# for i in range(num_procs):
	# 	n = run_set_n + i*run_set_n*10
	# 	p = run_set_p + i*run_set_p*5
	# 	k = run_set_k + i*run_set_k*2
	# 	run_set.append([n, p, k, True])

	print("| # of rows | # of colums | k neighbors | approx. operations | time (s) | parallel | # of processes | accuracy")
	
	run_knn(816, 2, 3, False, 1)
	run_knn(816, 2, 3, True, 50)

	# rows
	run_set1 = [ [10, 4, 10], [100, 4, 10], [1000, 4, 10], [10000, 4, 10] ]
	for row in run_set1:
		run_knn(row[0], row[1], row[2], False, 1)
		run_knn(row[0], row[1], row[2], True, 50)

	# columns
	run_set2 = [ [1000, 4, 10], [1000, 40, 10], [1000, 400, 10], [1000, 4000, 10] ]
	for row in run_set2:
		run_knn(row[0], row[1], row[2], False, 1)
		run_knn(row[0], row[1], row[2], True, 50)

	# k neighbors
	run_set3 = [ [1000, 4, 10], [1000, 4, 100], [1000, 4, 1000], [1000, 4, 10000] ]
	for row in run_set3:
		run_knn(row[0], row[1], row[2], False, 1)
		run_knn(row[0], row[1], row[2], True, 50)

	# processes
	run_set4 = [ [1000, 4, 10, 16], [1000, 4, 10, 32], [1000, 4, 10, 64], [1000, 4, 10, 128] ]
	for row in run_set4:
		run_knn(row[0], row[1], row[2], False, 1)
		run_knn(row[0], row[1], row[2], True, row[3])


