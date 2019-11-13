from knn import Knn
import numpy as np
import datetime
import math


def run_knn(n, p, k):

	#initialize random data here
	if n > 0 and p > 0:
		X_train = np.random.rand(int(n*.8), p)
		y_train = np.concatenate((np.zeros(int((n/2)*.8)), np.ones(int((n/2)*.8))))
		X_dev = np.random.rand(int(n*.2), p)
			
	one = str(datetime.datetime.now().time()).split(":")

	knn = Knn(k, X_dev, X_train, y_train)

	two = str(datetime.datetime.now().time()).split(":")

	# KNN cost = run() --> get_neighbors() + get_majority_vote()
	# = (for each X_dev example) * { [get its neighbors] + [then get the top vote] }
	# = (# of X_dev examples) * ...
	# ...{ [ (# of X_train * ) + (sort # of X_train) + (k)] + [ (k*2) + (k) ] }
	len_X_train = len(X_train)
	cost = len(X_dev) * ( ( (len_X_train*(2 + 4 + p*2)) + (len_X_train * math.log(len_X_train)) + (k) ) + (k*2 + k) )

	time_diff = (float(two[0]) - float(one[0]))*3600 + (float(two[1]) - float(one[1]))*60 + (float(two[2]) - float(one[2]))

	print(str(n) + ", " + str(p) + ", " + str(k) + ", " + str(int(cost)) + ", " + str(time_diff))


if __name__ == "__main__":

	print("| # of rows | # of colums | k neighbors | approx. operations | time (s) |")
	
	run_set1 = [ [10, 2, 5], [100, 2, 5], [1000, 2, 5], [10000, 2, 5]]
	# run_set1 = [ [10, 2, 5], [100, 2, 5], [1000, 2, 5], [10000, 2, 5], [100000, 2, 5], [1000000, 2, 5], [10000000, 2, 5]]

	for row in run_set1:
		run_knn(row[0], row[1], row[2])


	run_set2 = [ [1000, 2, 5], [1000, 20, 5], [1000, 200, 5], [1000, 2000, 5]]
	# run_set2 = [ [1000, 2, 5], [1000, 4, 5], [1000, 8, 5], [1000, 16, 5], [100000, 32, 5], [1000000, 64, 5], [10000000, 128, 5]]

	for row in run_set2:
		run_knn(row[0], row[1], row[2])


	run_set3 = [ [1000, 2, 5], [1000, 2, 25], [1000, 2, 125], [1000, 2, 625]]
	# run_set3 = [ [1000, 2, 5], [1000, 2, 10], [1000, 2, 20], [1000, 2, 40], [1000, 2, 80], [1000, 2, 160], [1000, 2, 320]]

	for row in run_set3:
		run_knn(row[0], row[1], row[2])