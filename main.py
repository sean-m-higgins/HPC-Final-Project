from knn import Knn


if __name__ == "__main__":
	import datetime
	print(datetime.datetime.now())

	#initialize random data here


	#-----------------------------------------------------------------------
	# KNN

	k = 8

	knn = Knn(k, X_dev, X_train, y_train)

	print("Done.")
	print(datetime.datetime.now())