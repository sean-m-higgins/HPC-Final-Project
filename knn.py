from math import *
import operator
import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe
import time


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
		self.check = True
		self.run()

	def run(self):
		index = 0 
		for instance in self.X_dev:
			if index == 3:
				self.check = False
			index += 1
			self.neighbors = []
			self.get_neighbors(instance)
			self.predictions.append(self.get_majority_vote())

	#from https://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
	def euclidean_distance(self, row_one, row_two):
	    """ distance = sqrt( sum( (differences between Ai and Bi)(squared) ) ) """
	    return sqrt(sum((pow(a-b, 2)) for a, b in zip(row_one, row_two)))
	  
	def get_neighbors(self, test_instance):
	  	""" get distances, sort, and return the k nearest neighbors """
	  	pipe_list = []
	  	for i in range(self.num_procs):
	  		parent_conn, child_conn = Pipe()
	  		pipe_list.append([parent_conn, child_conn])

	  	if self.check:
	  		start = time.time()

	  	proc_list = []
	  	for i in range(1, self.num_procs+1):
	  		chunk = int(len(self.X_train)/self.num_procs)  #have chunk size as param and change num_procs based on chunk size
	  		x_slice = self.X_train[(i-1)*chunk:i*chunk]
	  		y_slice = self.y_train[(i-1)*chunk:i*chunk]

	  		p = Process(target=self.get_distances, args=(test_instance, x_slice, y_slice, pipe_list[i-1][1]))
	  		proc_list.append(p)
	  		p.start()

	  	if self.check:
		  	end = time.time()
		  	print("proc time diff: " + str(end - start))

	  	all_distances = []
	  	for p, pipe in zip(proc_list, pipe_list):  # check status of worker/ recieved message in pipe # what if next pipe is not ready. will it wait? way to do any pipe from list? maybe first check if pipe is ready or if pipe in list of finished pipes
	  		p.join()
	  		next_arr = pipe[0].recv()
	  		for item in next_arr:
	  			all_distances.append(item)

	  	if self.check:
		  	end = time.time()
		  	print("proc and assign time diff: " + str(end - start))

	  	all_distances.sort(key=operator.itemgetter(1)) 

	  	for i in range(self.k):
	  		self.neighbors.append(all_distances[i][2])

	def get_distances(self, test_instance, X, y, pipe):
		new_distances = []

		if self.check:
			start = time.time()
		
		for X_new, y_new in zip(X, y): 
			dist = self.euclidean_distance(test_instance, X_new)
			new_distances.append([X_new, dist, y_new])

		if self.check:
			end = time.time()
			print("get_dist time diff: " + str(end - start))

		pipe.send(new_distances)  

	def get_majority_vote(self):
	  	""" return the vote with the highest count """
	  	class_votes = {}
	  	for vote in self.neighbors:
	  		class_votes.setdefault(vote, 0)
	  		class_votes[vote] += 1
	  	sorted_votes = sorted(class_votes.items(), key=lambda kv: kv[1], reverse=True)
	  	return sorted_votes[0][0]
