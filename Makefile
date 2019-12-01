CC=g++
CFLAGS=-I.

TARGET=knn

compile: Knn.cpp
	$(CC) -o $(TARGET) Knn.cpp -std=c++11

clean:
	$(RM) $(TARGET)