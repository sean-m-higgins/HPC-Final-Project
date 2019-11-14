CC=gcc
CFLAGS=-I.

TARGET=knn

compile: Knn.cpp
	$(CC) -o $(TARGET) Knn.cpp

clean:
	$(RM) $(TARGET)