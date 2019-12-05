from math import *
import operator
import numpy as np
import multiprocessing
from multiprocessing import Pool, Process, Queue, Pipe


class Knn:

	def __init__(self, k, X_dev, X_train, y_train):
		self.k = k
		self.X_dev = np.asarray(X_dev)
		self.X_train = np.asarray(X_train)
		self.y_train = np.asarray(y_train)
		self.neighbors = []
		self.predictions = []
		self.run()

	def run(self):
		for instance in self.X_dev:
			self.neighbors = []
			self.get_neighbors(instance)
			self.predictions.append(self.get_majority_vote())

	#from https://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
	def euclidean_distance(self, row_one, row_two):
	    """ distance = sqrt( sum( (differences between Ai and Bi)(squared) ) ) """
	    return sqrt(sum((pow(a-b, 2)) for a, b in zip(row_one, row_two)))
	  
	def get_neighbors(self, test_instance):
	  	""" get distances, sort, and retrun the k nearest neighbors """
	  	distances = []
	  	labels_index = 0
	  	for instance in self.X_train:
	  		dist = self.euclidean_distance(test_instance, instance)
	  		distances.append([instance, dist, self.y_train[labels_index]])
	  		labels_index += 1
	  	distances.sort(key=operator.itemgetter(1))
	  	for i in range(self.k):
	  		self.neighbors.append(distances[i][2])

	def get_majority_vote(self):
	  	""" return the vote with the highest count """
	  	class_votes = {}
	  	for vote in self.neighbors:
	  		class_votes.setdefault(vote, 0)
	  		class_votes[vote] += 1
	  	sorted_votes = sorted(class_votes.items(), key=lambda kv: kv[1], reverse=True)
	  	return sorted_votes[0][0]

class KnnParallel:

	def __init__(self, k, X_dev, X_train, y_train, num_procs):
		self.k = k
		self.X_dev = np.asarray(X_dev)
		self.X_train = np.asarray(X_train)
		self.y_train = y_train
		self.neighbors = []
		self.predictions = []
		self.num_procs = num_procs
		self.run()

	def run(self):
		for instance in self.X_dev:
			self.neighbors = []
			self.get_neighbors(instance)
			self.predictions.append(self.get_majority_vote())

	#from https://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
	def euclidean_distance(self, row_one, row_two):
	    """ distance = sqrt( sum( (differences between Ai and Bi)(squared) ) ) """
	    return sqrt(sum((pow(a-b, 2)) for a, b in zip(row_one, row_two)))
	  
	def get_neighbors(self, test_instance):
	  	""" get distances, sort, and return the k nearest neighbors """
	  	# distances=Queue()
	  	p_list = []

	  	# parent_conn, child_conn = Pipe()
	  	
	  	# for loop to create 50 pipes?
	  	pipe_list = []
	  	for i in range(self.num_procs):
	  		parent_conn, child_conn = Pipe()
	  		pipe_list.append([parent_conn, child_conn])

	  	for i in range(1, self.num_procs+1):  #TODO
	  		chunk = int(len(self.X_train)/self.num_procs)

	  		x_slice = self.X_train[(i-1)*chunk:i*chunk]
	  		y_slice = self.y_train[(i-1)*chunk:i*chunk]

	  		# create the process
	  		p = Process(target=self.get_distances, args=(test_instance, x_slice, y_slice, pipe_list[i-1][1]))
	  		p_list.append(p)
	  		p.start()

	  	all_distances = []  #TODO better way to do this?
	  	# print("p_list: " + str(len(p_list)))
	  	for p, conn in zip(p_list, pipe_list):
	  		p.join()
	  		next_arr = conn[0].recv()
	  		# print("Done")
	  		for item in next_arr:
	  			all_distances.append(item)  #TODO better way to do this?


	  	# collect the individual distances
	  	# top_distances = []
	  	# print(distances.qsize())	
	  	# # for i in range(distances.qsize()+1):
	  	# for i in range(50):
	  	# 	new_distances = distances.get()
	  	# 	for item in new_distances:
	  	# 		top_distances.append(item)
	  	# print(distances.qsize())		
	  	# print(len(all_distances))
	  	all_distances.sort(key=operator.itemgetter(1))

	  	for i in range(self.k):
	  		self.neighbors.append(all_distances[i][2])

	def get_distances(self, test_instance, X, y, conn):
		new_distances = []
		for X_new, y_new in zip(X, y):
			dist = self.euclidean_distance(test_instance, X_new)
			new_distances.append([X_new, dist, y_new])
		conn.send(new_distances)
		# distances.put(new_distances)

	def get_majority_vote(self):
	  	""" return the vote with the highest count """
	  	class_votes = {}
	  	for vote in self.neighbors:
	  		class_votes.setdefault(vote, 0)
	  		class_votes[vote] += 1
	  	sorted_votes = sorted(class_votes.items(), key=lambda kv: kv[1], reverse=True)
	  	return sorted_votes[0][0]
