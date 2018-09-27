all: svm_main.exec
CC=g++
TARGET = svm_main.exec
LIBS=-pthread
CFLAGS=-Wall -g -fpermissive -std=c++11
OBJS=svm_main.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

svm_main.o: svm_main.cpp
	$(CC) $(CFLAGS) -c svm_main.cpp

clean:
	rm -rf *.o  *~
