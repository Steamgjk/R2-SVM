//
//  svm_main.cpp
//
//  Created by Jinkun Geng on 18/09/26.
//  Copyright (c) 2016年 Jinkun Geng. All rights reserved.
//

#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <cmath>
#include <time.h>
#include <vector>
#include <list>
#include <thread>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sys/time.h>
#include <map>
using namespace std;
#define DataSet "Iris.data"
#define DIM 4

struct SamplePoint
{
	double* xptr;
	int y;
};
std::vector<SamplePoint*> DS;

double* Wptr;
double* alphas;
double b;
int C = 100; //penalty
double tolerance = 0.001;
double eps = 1e-3; // convergence condition
int two_sigma_squared = 100; //RBF(Radial-Basis Function)核函数中的参数。sigma==(10/2)^1/2。
double error_cache[end_support_i];//存放non-bound样本误差 ??



void ReadData();
void SMO();
void InitPara();

void ReadData()
{
	ifstream ifs(DataSet);
	if (!ifs.is_open())
	{
		printf("fail-LoadD4 to open %s\n", DataSet );
		exit(-1);
	}
	char ch;
	int cnt = 0;
	while (!ifs.eof())
	{
		SamplePoint* sp = (SamplePoint*)malloc(sizeof(SamplePoint));
		sp->xptr = (double*)malloc(sizeof(double) * DIM);
		for (int i = 0; i < DIM; i++)
		{
			ifs >> (sp->xptr[i]);
			ifs >> ch;
		}
		ifs >> (sp->y);
		if (cnt % 10 == 0)
		{
			printf("cnt = %d\n", cnt );
		}
	}

}
void InitPara()
{
	Wptr = (double*)malloc(sizeof(double) * DIM);
	alphas = (double*)malloc(sizeof(double) * DIM);
	for (int i = 0; i < DIM; i++)
	{
		Wptr[i] = drand48();
		alphas[i] = drand48();
	}
}
void SMO()
{

}
int main(int argc, const char * argv[])
{
	printf("Hello World\n");
	ReadData();
	printf("Data Loaded\n");
	InitPara();
}
