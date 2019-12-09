from knn import Knn, KnnParallel
import numpy as np
import time
import math
import multiprocessing
from multiprocessing import Process, Pool


def run_knn(n, p, k, parallel, num_procs):
	#initialize random data here
	if n > 0 and p > 0:
		np.random.seed(42)
		X_train = np.random.rand(int(n*.8), p)
		y_train = np.concatenate((np.zeros(int((n/2)*.8)), np.ones(int((n/2)*.8))))
		X_dev = np.random.rand(int(n*.2), p)
		y_dev = np.concatenate((np.zeros(int((n/2)*.2)), np.ones(int((n/2)*.2))))
			
	start = time.time()

	if parallel:
		knn = KnnParallel(k, X_dev, X_train, y_train, num_procs)
	else:
		knn = Knn(k, X_dev, X_train, y_train)

	end = time.time()

	predictions = knn.predictions
	correct = 0
	for pred, actual in zip(predictions, y_dev):
		if int(pred) == int(actual):
			correct += 1
	
	accuracy = correct/len(predictions)

	# KNN cost = run() --> get_neighbors() + get_majority_vote()
	# = (for each X_dev example) * { [get its neighbors] + [then get the top vote] }
	# = (# of X_dev examples) * ...
	# ...{ [ (# of X_train * ) + (sort # of X_train) + (k)] + [ (k*2) + (k) ] }
	len_X_train = len(X_train)
	cost = int( len(X_dev) * ( ( (len_X_train*(2 + 4 + p*2)) + \
		(len_X_train * math.log(len_X_train)) + (k) ) + (k*2 + k) ) )
	
	time_diff = end - start

	print("{}, {}, {}, {}, {}, {}, {}, {}".format(n, p, k, cost, 
		time_diff, parallel, num_procs, accuracy))


if __name__ == "__main__":

	print("| # of rows | # of colums | k neighbors | approx. operations | time (s) | parallel | # of processes | accuracy |")

	# rows
	run_set1 = [ [1000, 4, 5], [2000, 4, 5], [3000, 4, 5], [4000, 4, 5], [5000, 4, 5] ]
	# columns
	run_set2 = [ [1000, 4, 10], [1000, 40, 10], [1000, 400, 10], [1000, 4000, 10] ]#, [1000, 40000, 10] ]
	# k neighbors
	run_set3 = [ [1000, 4, 10], [1000, 4, 100], [1000, 4, 200], [1000, 4, 400], [1000, 4, 800] ]
	# processes
	run_set4 = [[2500, 4, 5, 8], [2500, 4, 5, 16], [2500, 4, 5, 32], [2500, 4, 5, 64], [2500, 4, 5, 128], [2500, 4, 5, 272] ]
	
	# # Serial
	# for row in run_set1:
	# 	run_knn(row[0], row[1], row[2], False, 1)

	# print("Done")

	# for row in run_set2:
	# 	run_knn(row[0], row[1], row[2], False, 1)

	# run_knn(1000, 40000, 10, False, 1)
	# print("Done")

	# for row in run_set3:
	# 	run_knn(row[0], row[1], row[2], False, 1)

	# print("Done")

	# # Parallel
	# for row in run_set1:
	# 	run_knn(row[0], row[1], row[2], True, 50)

	# print("Done")

	# for row in run_set2:
	# 	run_knn(row[0], row[1], row[2], True, 50)

	# print("Done")

	# for row in run_set3:
	# 	run_knn(row[0], row[1], row[2], True, 50)

	# print("Done")

	for row in run_set4:
		run_knn(row[0], row[1], row[2], True, row[3])

	print("Done")

