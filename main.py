from knn import Knn, KnnParallel
import numpy as np
import datetime
import math
import multiprocessing
from multiprocessing import Process, Pool


def run_knn(n, p, k, parallel):
	#initialize random data here
	if n > 0 and p > 0:
		np.random.seed(42)
		X_train = np.random.rand(int(n*.8), p)
		y_train = np.concatenate((np.zeros(int((n/2)*.8)), np.ones(int((n/2)*.8))))
		X_dev = np.random.rand(int(n*.2), p)
			
	one = str(datetime.datetime.now().time()).split(":")

	if parallel:
		knn = KnnParallel(k, X_dev, X_train, y_train)
	else:
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

	run_set_n = int(input("Enter Initial # of rows: "))
	run_set_p = int(input("Enter Initial # of colums: "))
	run_set_k = int(input("Enter Initial k neighbors: "))
	# get the number of CPUs 
	num_procs = multiprocessing.cpu_count()
	print('You have {0:1d} CPUs'.format(num_procs))

	run_set = []
	for i in range(num_procs):
		n = run_set_n + i*run_set_n*10
		p = run_set_p + i*run_set_p*5
		k = run_set_k + i*run_set_k*2
		run_set.append([n, p, k, True])

	print("| # of rows | # of colums | k neighbors | approx. operations | time (s) |")
	# print("Serial:")
	# for row in run_set:
	# 	run_knn(row[0], row[1], row[2], False)
	run_knn(816, 2, 3, False)
	run_knn(816, 2, 3, True)

	# print("Parallel:")
	# # Create the processes
	# p_list=[]
	# for i in range(1,num_procs+1):
	# 	cur_n = run_set[i-1][0]
	# 	cur_p = run_set[i-1][1]
	# 	cur_k = run_set[i-1][2]
	# 	p = Process(target=run_knn, name='Process'+str(i), args=(cur_n, cur_p, cur_k, True,))
	# 	p_list.append(p)
	# 	p.start()
	# 	print('Process:: ', p.name, 'Was assigned PID:: ', p.pid)

	# # Wait for all the processes to finish
	# for p in p_list:
	# 	p.join()


	# # Create the worker pool
	# pool = Pool(processes=num_procs)

	# # parallel map
	# pool.map(run_knn, arguments)





	# run_set1 = [ [10, 2, 5], [100, 2, 5], [1000, 2, 5], [10000, 2, 5] ]
	# # run_set1 = [ [10, 2, 5], [100, 2, 5], [1000, 2, 5], [10000, 2, 5], [100000, 2, 5], [1000000, 2, 5], [10000000, 2, 5]]

	# for row in run_set1:
	# 	run_knn(row[0], row[1], row[2])


	# run_set2 = [ [1000, 2, 5], [1000, 20, 5], [1000, 200, 5], [1000, 2000, 5] ]
	# # run_set2 = [ [1000, 2, 5], [1000, 4, 5], [1000, 8, 5], [1000, 16, 5], [100000, 32, 5], [1000000, 64, 5], [10000000, 128, 5]]

	# for row in run_set2:
	# 	run_knn(row[0], row[1], row[2])


	# run_set3 = [ [1000, 2, 10], [1000, 2, 100], [1000, 2, 1000], [1000, 2, 10000] ]
	# # run_set3 = [ [1000, 2, 5], [1000, 2, 10], [1000, 2, 20], [1000, 2, 40], [1000, 2, 80], [1000, 2, 160], [1000, 2, 320]]

	# for row in run_set3:
	# 	run_knn(row[0], row[1], row[2])