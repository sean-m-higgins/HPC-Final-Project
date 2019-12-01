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

	def __init__(self, k, X_dev, X_train, y_train):
		self.k = k
		self.X_dev = np.asarray(X_dev)
		self.X_train = np.asarray(X_train)
		self.y_train = y_train
		self.neighbors = []
		self.predictions = []
		self.num_procs = multiprocessing.cpu_count()
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

	  	# Create the Queue which will have the partial products
	  	distances=Queue()

	  	for i in range(1, self.num_procs+1):
	  		# A pair of queues per process for the two arrays
	  		xq = Queue()
	  		yq = Queue()

	  		chunk = int(len(self.X_train)/self.num_procs)
	  		# push the chunks into the queue
	  		xq.put(self.X_train[(i-1)*chunk:i*chunk])
	  		yq.put(self.y_train[(i-1)*chunk:i*chunk])

	  		# create the process
	  		p = Process(target=self.get_distances, args=(test_instance, xq, yq, distances))
	  		p.start()
	  		p.join()

	  	# collect the individual distances
	  	top_distances = []
	  	for i in range(distances.qsize()):
	  		top_distances.append(distances.get())
	  
	  	top_distances.sort(key=operator.itemgetter(1))
	  	for i in range(self.k):
	  		self.neighbors.append([top_distances[i][0], top_distances[i][2]])

	def get_distances(self, test_instance, Xqueue, Yqueue, distances):
		x_instances = Xqueue.get()
		y_instances = Yqueue.get()
		for X, y in zip(x_instances, y_instances):
			dist = self.euclidean_distance(test_instance, X)
			distances.put([X, dist, y])

	def get_majority_vote(self):
	  	""" return the vote with the highest count """
	  	class_votes = {}
	  	for row in self.neighbors:
	  		vote = row[1]
	  		class_votes.setdefault(vote, 0)
	  		class_votes[vote] += 1
	  	sorted_votes = sorted(class_votes.items(), key=lambda kv: kv[1], reverse=True)
	  	return sorted_votes[0][0]
