CXX ?= g++
CFLAGS = -Wall -std=c++11 -Wconversion -fPIC
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale my-svm-train

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm -lpthread
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm -lpthread
my-svm-train: my-svm-train.c svm.o
	$(CXX) $(CFLAGS) my-svm-train.c svm.o -o my-svm-train -lm -lpthread
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale -lpthread
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o my-svm-train svm-train svm-predict svm-scale libsvm.so.$(SHVER)