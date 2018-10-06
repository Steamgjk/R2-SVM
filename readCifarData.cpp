
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
#define DataSet "cifar-10-batches-bin/data_batch_"
#define OutPutSet "cifar-10-batches-bin/cifar_output"

#define TestSet "cifar-10-batches-bin/test_batch.bin"
#define OutPutTest "cifar-10-batches-bin/cifar_test"

int main(int argc, const char * argv[])
{
	char label;
	char r_val[1024];
	char g_val[1024];
	char b_val[1024];

	ofstream outputS(OutPutSet, ios::trunc);
	//ofstream outputS(OutPutTest, ios::trunc);
	char fn[100];
	int cnt = 0;
	int tmp = 0;
	for (int j = 1; j <= 5; j++)
	{
		sprintf(fn, "%s%d.bin", DataSet, j);
		ifstream inputS(fn, ios::in | ios::binary );
		//ifstream inputS(TestSet, ios::in | ios::binary );
		while (inputS.peek() != EOF)
		{
			inputS.read(&label, 1);

			//outputS << label + 0 << "\t";
			//outputS << unsigned(label) << "\t";
			for (int i = 0; i < 1024; i++)
			{
				inputS.read(&(r_val[i]), 1);
				tmp = r_val[i];
				if (tmp < 0)
				{
					tmp += 256;
				}
				outputS << tmp << "\t";
				//outputS << r_val[i] + 0 << "\t";
				//outputS << unsigned(r_val[i]) << "\t";
			}
			for (int i = 0; i < 1024; i++)
			{
				inputS.read(&(g_val[i]), 1);
				tmp = g_val[i];
				if (tmp < 0)
				{
					tmp += 256;
				}
				outputS << tmp << "\t";
				//outputS << g_val[i] + 0 << "\t";
				//outputS << unsigned(g_val[i]) << "\t";
			}
			for (int i = 0; i < 1024; i++)
			{
				inputS.read(&(b_val[i]), 1);
				tmp = b_val[i];
				if (tmp < 0)
				{
					tmp += 256;
				}
				outputS << tmp << "\t";
				//outputS << b_val[i] + 0 << "\t";
				//outputS << unsigned(b_val[i]) << "\t";
			}
			if (label < 5)
			{
				outputS << 1;
			}
			else
			{
				outputS << -1;
			}
			outputS << endl;
			cnt++;
			if (cnt % 100 == 0)
			{
				printf("cnt=%d\n", cnt );
			}
			//getchar();

		}
	}


}
