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
//#define TrainData "train_data"
//#define TestData "test_data"
//#define DIM 4

#define TrainData "cifar-10-batches-bin/cifar_output"
#define TestData "cifar-10-batches-bin/cifar_test"
#define DIM 3072

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
double b = 0.0;
int C = 100; //penalty
double tolerance = 0.001;
double eps = 1e-3; // convergence condition
int two_sigma_squared = 100; //RBF(Radial-Basis Function)核函数中的参数。sigma==(10/2)^1/2。
int use_linear_kernel = 1;
std::vector<double> W;
std::vector<double> error_cache;//存放non-bound样本误差



void ReadData();
void SMO();
void InitPara();
int examine_example(int k, int is_linear = 1);
int take_step(int i1, int i2, int is_linear = 1);
double dot_product(int i1, int i2);
double dot_product(vector<double>& vec_1, double*vec_2);
double kernel(int i1, int i2, int is_linear = 1);
double learned_func(int k, int is_linear = 1);
void PrintModel();
void SVM_Test();

void SVM_Test()
{
	double ee = 0.0;
	printf("DS sz = %ld\n", DS.size() );
	for (int i = train_cnt; i < train_cnt + test_cnt; i++)
	{
		if ((DS[i]->y) * learned_func(i, use_linear_kernel) < 0)
		{
			ee += 1.0;
		}
		printf("%d:%lf\n", DS[i]->y, learned_func(i, use_linear_kernel)  );
	}
	printf("ee=%lf ratio=%lf\n", ee, ee / test_cnt );
}
void PrintModel()
{
	for (int i = 0; i < train_cnt; i++)
	{
		if (alphas[i] > 0.0)
		{
			printf("[%d]:%lf\n", i, alphas[i]);
		}
	}
	printf("b=%lf\n", b );
}
double dot_product(int i1, int i2)
{
	double res = 0.0;
	for (int i = 0; i < DIM; i++)
	{
		res += (DS[i1]->xptr[i]) * (DS[i2]->xptr[i]);
	}
	return res;
}
double dot_product(vector<double>& vec_1, double*vec_2)
{
	double res = 0.0;
	for (int i = 0; i < DIM; i++)
	{
		res += (vec_1[i]) * (vec_2[i]);
	}
	return res;
}
double kernel(int i1, int i2, int is_linear)
{
	if (is_linear)
	{
		return dot_product(i1, i2);
	}
	else
	{
		double r = 0.0;
		for (int i = 0; i < DIM; i++)
		{
			r += ((DS[i1]->xptr[i]) - (DS[i2]->xptr[i])) * ((DS[i1]->xptr[i]) - (DS[i2]->xptr[i]));
		}
		r /= (-2 * two_sigma_squared);
		r = exp(r);
		return r;
	}
}

double learned_func(int k, int is_linear)
{
	float s = 0.0;
	if (is_linear)
	{
		s = dot_product(W, DS[k]->xptr);
	}
	else
	{
		for (int i = 0; i < train_cnt; i++)
		{
			if (alphas[i] > 0)
			{
				s += alphas[i] * DS[i]->y * kernel(i, k);
			}
		}
	}
	s -= b;
	return s;
}


//If it makes posiive progress, then return 1, else return 0
int take_step(int i1, int i2, int is_linear)
{
	int y1 = 0;
	int y2 = 0;
	int s = 0;
	double alpha1 = 0.0; //old values of multipliers
	double alpha2 = 0.0;
	double a1 = 0.0; // new values of multipliers
	double a2 = 0.0;
	double e1 = 0.0;
	double e2 = 0.0;
	double low = 0.0;
	double high = 0.0;
	double k11 = 0.0;
	double k22 = 0.0;
	double k12 = 0.0;
	double eta = 0.0;
	double low_obj = 0.0;
	double high_obj = 0.0;
	if (i1 == i2)
	{
		return 0;
	}

	alpha1 = alphas[i1];
	alpha2 = alphas[i2];
	y1 = DS[i1]->y;
	y2 = DS[i2]->y;
	//non-bound has been cached
	if (alpha1 > 0 && alpha1 < C)
	{
		e1 = error_cache[i1];
	}
	else
	{
		e1 = learned_func(i1, is_linear) - y1;
	}
	if (alpha2 > 0 && alpha2 < C)
	{
		e2 = error_cache[i2];
	}
	else
	{
		e2 = learned_func(i2, is_linear) - y2;
	}
	s = y1 * y2;
	if (y1 == y2)
	{
		low = max(0.0, alpha1 + alpha2 - C);
		high = min(C * 1.0, alpha1 + alpha2);
	}
	else
	{
		low = max(0.0, alpha2 - alpha1);
		high = min(C * 1.0, C + alpha2 - alpha1);
	}

	if (fabs(low - high) < 1e-6)
	{
		return 0;
	}

	k11 = kernel(i1, i1, is_linear);
	k12 = kernel(i1, i2, is_linear);
	k22 = kernel(i2, i2, is_linear);
	eta = 2 * k12 - k11 - k22;

	if (eta < 0)
	{
		a2 = alpha2 + y2 * (e2 - e1) / eta;
		if (a2 < low)
		{
			a2 = low;
		}
		else if (a2 > high)
		{
			a2 = high;
		}
	}
	else  // if eta == 0
	{
///???
		double c1 = eta / 2.0;
		double c2 = y2 * (e1 - e2) - eta * alpha2;
		low_obj = c1 * low * low + c2 * low;
		high_obj = c1 * high * high + c2 * high;
		if (low_obj > high_obj + eps)
		{
			a2 = low;
		}
		else if (low_obj < high_obj - eps)
		{
			a2 = high;
		}
		else
		{
			a2 = alpha2;
		}
	}

	if (fabs(a2 - alpha2) < eps * (a2 + alpha2 + eps))
	{
		return 0;
	}
	a1 = alpha1 - s * (a2 - alpha2);
	if (a1 < 0)
	{
		a2 += s * a1;
		a1 = 0;
	}
	else if (a1 > C)
	{
		double t = a1 - C;
		a2 += s * t;
		a1 = C;
	}

	double b1 = 0.0;
	double b2 = 0.0;
	double bnew = 0.0;
	if (a1 > 0 && a1 < C)
	{
		bnew = b + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
	}
	else if (a2 > 0 && a2 < C)
	{
		bnew = b + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
	}
	else
	{
		b1 = b + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
		b2 = b + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
		bnew = (b1 + b2) / 2.0;
	}

	double delta_b = bnew - b;
	b = bnew;

	double t1 = y1 * (a1 - alpha1);
	double t2 = y2 * (a2 - alpha2);

	if (is_linear)
	{
		for (int i = 0; i < DIM; i++)
		{
			W[i] += t1 * DS[i1]->xptr[i] + t2 * DS[i2]->xptr[i];
		}
	}

	for (int i = 0; i < train_cnt; i++)
	{
		if (alphas[i] > 0 && alphas[i] < C)
		{
			error_cache[i] += t1 * kernel(i1, i, is_linear) + t2 * kernel(i2, i, is_linear) - delta_b;
		}
	}

	error_cache[i1] = 0.0;
	error_cache[i2] = 0.0;

	alphas[i1] = a1;
	alphas[i2] = a2;
	return 1;
}


int examine_example(int i1, int is_linear)
{
	double y1 = 0.0;
	double alpha1 = 0.0;
	double e1 = 0.0;
	double r1 = 0.0;

	y1 = DS[i1]->y;
	alpha1 = alphas[i1];
	if (alpha1 > 0 && alpha1 < C)
	{
		e1 = error_cache[i1];
	}
	else
	{
		e1 = learned_func(i1, is_linear) - y1;
	}

	r1 = y1 * e1;
	if ((r1 < -tolerance && alpha1 < C) || (r1 > tolerance && alpha1 > 0))
	{
		int k0 = 0;
		int k = 0;
		int i2 = -1;
		double tmax = 0.0;
		for (i2 = -1, tmax = 0, k = 0; k < train_cnt; k++)
		{
			if (alphas[k] > 0 && alphas[k] < C)
			{
				double e2 = 0.0;
				double temp = 0.0;
				e2 = error_cache[k];
				temp = fabs(e1 - e2);
				if (temp > tmax)
				{
					tmax = temp;
					i2 = k;
				}
			}

		}
		if (i2 >= 0)
		{
			if (take_step(i1, i2, is_linear))
			{
				return 1;
			}
		}

		for (k0 = (int)(drand48() * train_cnt), k = k0; k < train_cnt + k0; k++)
		{
			i2 = k % train_cnt;
			if (alphas[i2] > 0 && alphas[i2] < C)
			{
				if (take_step(i1, i2, is_linear))
				{
					return 1;
				}
			}
		}
		for (k0 = (int)(drand48() * train_cnt), k = k0; k < train_cnt + k0; k++)
		{
			i2 = k % train_cnt;
			if (take_step(i1, i2, is_linear))
			{
				return 1;
			}
		}
	}
	return 0;
}

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
		if (train_cnt % 100 == 0)
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
		if (test_cnt % 100 == 0)
		{
			printf("test_cnt = %d\n", test_cnt );
		}
		DS.push_back(sp);
		test_cnt++;
	}
	//random_shuffle(DS.begin(), DS.end());
	int total_cnt = train_cnt + test_cnt;
	//train_cnt = total_cnt / 2;
	//test_cnt = total_cnt - train_cnt;
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
	W.resize(DIM, 0.0);
	alphas.resize(train_cnt, 0.0);
	error_cache.resize(train_cnt, 0.0);
	b = 0.0;
}

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
				num_changed += examine_example(k, use_linear_kernel);
			}
		}
		else
		{
			for (int k = 0; k < train_cnt; k++)
			{
				if (alphas[k] != 0 && alphas[k] != C)
				{
					num_changed += examine_example(k, use_linear_kernel);
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


//Alignment, the orignal algo does not have this
		for (int i = 0; i < train_cnt; i++)
		{
			if (alphas[i] < 1e-6)
			{
				alphas[i] = 0.0;
			}
		}
		printf("examine_all=%d  num_changed=%d\n", examine_all, num_changed );
		//PrintModel();
		//getchar();
	}

}

int main(int argc, const char * argv[])
{
	use_linear_kernel = 1;
	printf("Hello World\n");
	ReadData();
	printf("Data Loaded\n");
	return;
	InitPara();
	printf("InitPara Finished train_cnt=%d\n", train_cnt);
	SMO();
	PrintModel();
	SVM_Test();
}
