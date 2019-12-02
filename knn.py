from math import *
import operator
import numpy as np
import multiprocessing
from multiprocessing import Pool, Process, Queue


class Knn:

	def __init__(self, k, X_dev, X_train, y_train):
		self.k = k
		self.X_dev = np.asarray(X_dev)
		self.X_train = np.asarray(X_train)
		self.y_train = y_train
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
	  		self.neighbors.append([distances[i][0], distances[i][2]])

	def get_majority_vote(self):
	  	""" return the vote with the highest count """
	  	class_votes = {}
	  	for row in self.neighbors:
	  		vote = row[1]
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
	  	""" get distances, sort, and retrun the k nearest neighbors """
	  	distances=Queue()
	  	p_list = []

	  	for i in range(1, self.num_procs):
	  		chunk = int(len(self.X_train)/self.num_procs)

	  		x_slice = self.X_train[(i-1)*chunk:i*chunk]
	  		y_slice = self.y_train[(i-1)*chunk:i*chunk]

	  		# create the process
	  		p = Process(target=self.get_distances, args=(test_instance, x_slice, y_slice, distances))
	  		p_list.append(p)
	  		p.start()

	  	for p in p_list:
	  		p.join

	  	# collect the individual distances
	  	top_distances = []
	  	for i in range(distances.qsize()):
	  		new_distances = distances.get()
	  		for item in new_distances:
	  			top_distances.append(item)
	  
	  	top_distances.sort(key=operator.itemgetter(1))
	  	for i in range(self.k):
	  		self.neighbors.append([top_distances[i][0], top_distances[i][2]])

	def get_distances(self, test_instance, X, y, distances):
		new_distances = []
		for X_new, y_new in zip(X, y):
			dist = self.euclidean_distance(test_instance, X_new)
			new_distances.append([X_new, dist, y_new])
		distances.put(new_distances)

	def get_majority_vote(self):
	  	""" return the vote with the highest count """
	  	class_votes = {}
	  	for row in self.neighbors:
	  		vote = row[1]
	  		class_votes.setdefault(vote, 0)
	  		class_votes[vote] += 1
	  	sorted_votes = sorted(class_votes.items(), key=lambda kv: kv[1], reverse=True)
	  	return sorted_votes[0][0]
