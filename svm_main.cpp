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
#include <random>

using namespace std;
#define DataSet "Iris.data"
#define TrainData "train_data"
#define TestData "test_data"
#define DIM 4

struct SamplePoint
{
	double* xptr;
	int y;
	void Output()
	{
		for (int i = 0; i < DIM; i++)
		{
			printf("%lf ", xptr[i] );
		}
		printf("%d\n", y );
	}
};
std::vector<SamplePoint*> DS;
std::vector<SamplePoint*> TestDS;
int train_cnt = 0;
int test_cnt = 0;

std::vector<double> alphas;
double b;
int C = 100; //penalty
double tolerance = 0.001;
double eps = 1e-3; // convergence condition
int two_sigma_squared = 100; //RBF(Radial-Basis Function)核函数中的参数。sigma==(10/2)^1/2。
std::vector<double> error_cache;//存放non-bound样本误差



void ReadData();
void SMO();
void InitPara();
int examine_example(int k);
double kernel(int i, int j);

void ReadData()
{
	ifstream ifs(TrainData);
	if (!ifs.is_open())
	{
		printf("fail-LoadD4 to open %s\n", TrainData );
		exit(-1);
	}

	while (!ifs.eof())
	{
		SamplePoint* sp = (SamplePoint*)malloc(sizeof(SamplePoint));
		sp->xptr = (double*)malloc(sizeof(double) * DIM);
		for (int i = 0; i < DIM; i++)
		{
			ifs >> (sp->xptr[i]);
		}
		ifs >> (sp->y);
		if (train_cnt % 10 == 0)
		{
			printf("train_cnt = %d\n", train_cnt );
		}
		DS.push_back(sp);
		train_cnt++;
		//sp->Output();
		//getchar();
	}

	ifstream ifs_test(TestData);
	if (!ifs_test.is_open())
	{
		printf("fail-LoadD4 to open %s\n", TestData );
		exit(-1);
	}
	while (!ifs_test.eof())
	{
		SamplePoint* sp = (SamplePoint*)malloc(sizeof(SamplePoint));
		sp->xptr = (double*)malloc(sizeof(double) * DIM);
		for (int i = 0; i < DIM; i++)
		{
			ifs_test >> (sp->xptr[i]);
		}
		ifs_test >> (sp->y);
		if (test_cnt % 10 == 0)
		{
			printf("test_cnt = %d\n", test_cnt );
		}
		TestDS.push_back(sp);
		test_cnt++;
	}
	printf("Load Finished train=%d  test=%d\n", train_cnt, test_cnt );


	/*
	ofstream traingf(TrainData);
	ofstream testf(TestData);
	for (int i = 0; i < DS.size(); i++)
	{
		if (rand() % 100 >= 10)
		{
			for (int j = 0; j < DIM; j++)
			{
				traingf << (DS[i]->xptr[j]) << " ";
			}
			traingf << (DS[i]->y);
			traingf << endl;
		}
		else
		{
			for (int j = 0; j < DIM; j++)
			{
				testf << (DS[i]->xptr[j]) << " ";
			}
			testf << (DS[i]->y);
			testf << endl;
		}
	}
	**/

}
void InitPara()
{
	alphas.resize(train_cnt, 0.0);
	error_cache.resize(train_cnt, 0.0);
	b = 0.0;
}
/*
void SMO()
{
	int num_changed = 0;
	int examine_all = 1;
	while (num_changed > 0 || examine_all)
	{
		num_changed = 0;
		if (examine_all)
		{
			for (int k = 0; k < train_cnt; k++)
			{
				num_changed += examine_example(k);
			}
		}
		else
		{
			for (int k = 0; k < train_cnt; k++)
			{
				if (alphas[k] != 0 && alphas[k] != C)
				{
					num_changed += examine_example(k);
				}
			}
		}
		//The first round examine all, but the following rounds examine the selected (0,c), unless no num changed from the selected set
		if (examine_all == 1)
		{
			examine_all = 0;
		}
		else if (num_changed == 0)
		{
			examine_all = 1;
		}

		double s = 0.0;
		double t = 0.0;
		for (int i = 0; i < train_cnt; i++)
		{
			s += alphas[i];
		}

		for (int i = 0; i < train_cnt; i++)
		{
			for (int j = 0; j < train_cnt; j++)
			{
				t += alphas[i] * alphas[j] * (DS[i]->y) * (DS[j]->y) * kernel(i, j);
			}
		}

		for (int i = 0; i < train_cnt; i++)
		{
			if (alphas[i] < 1e-6)
			{
				alphas[i] = 0.0;
			}
		}
	}

}
**/
int main(int argc, const char * argv[])
{
	printf("Hello World\n");
	ReadData();
	printf("Data Loaded\n");
	InitPara();
	printf("InitPara Finished\n");
	//SMO();
}
