#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <fstream>
#include <errno.h>
#include <thread>
#include "svm.h"
using namespace std;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	    "Usage: svm-train [options] training_set_file [model_file]\n"
	    "options:\n"
	    "-s svm_type : set type of SVM (default 0)\n"
	    "	0 -- C-SVC		(multi-class classification)\n"
	    "	1 -- nu-SVC		(multi-class classification)\n"
	    "	2 -- one-class SVM\n"
	    "	3 -- epsilon-SVR	(regression)\n"
	    "	4 -- nu-SVR		(regression)\n"
	    "-t kernel_type : set type of kernel function (default 2)\n"
	    "	0 -- linear: u'*v\n"
	    "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	    "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	    "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	    "	4 -- precomputed kernel (kernel values in training_set_file)\n"
	    "-d degree : set degree in kernel function (default 3)\n"
	    "-g gamma : set gamma in kernel function (default 1/num_features)\n"
	    "-r coef0 : set coef0 in kernel function (default 0)\n"
	    "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	    "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	    "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	    "-m cachesize : set cache memory size in MB (default 100)\n"
	    "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	    "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	    "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	    "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	    "-v n: n-fold cross validation mode\n"
	    "-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();
void r2_read_problem(char*buf, struct svm_problem& myprob);
void r2_read_problems(char*buf, struct svm_problem* myprobs, int prob_num);
void print_model(svm_model* model);
void sub_svm_train(int tid);

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

struct svm_problem* probs;
int prob_num = 4;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

int main(int argc, char **argv)
{
	printf("okkkk correlation=%d\n", cross_validation);
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	//read_problem(input_file_name);
	//read_problem wil be changed
	//read_problem(input_file_name);

	//buf=|sample_num(int)|dim_num(int)|label ... label(double)|features...(svm_node)
	int sample_num = 50000;
	int dim_num = 3072;
	size_t feature_num = (size_t)sample_num * dim_num;
	size_t buf_size = sizeof(int) + sizeof(int) + sample_num * sizeof(double) + feature_num * sizeof(struct svm_node);
	char* buf = Malloc(char, buf_size);
	ifstream inputS(input_file_name);
	if (!inputS.is_open())
	{
		printf("fail to open %s\n", input_file_name);
	}
	int* int_ptr = static_cast<int*>(static_cast<void*>(buf));
	int_ptr[0] = sample_num;
	int_ptr[1] = dim_num;
	double* double_ptr = static_cast<double*>(static_cast<void*>(int_ptr + 2));
	struct svm_node* svm_node_ptr = static_cast<struct svm_node*>(static_cast<void*>(double_ptr + sample_num));


	double ele; char ch; int key_pos; size_t base;
	int i, j;
	for (i = 0; i < sample_num; i++)
	{
		if (i % 1000 == 0)
		{
			printf("i=%d\n", i );
		}
		inputS >> ele;
		double_ptr[i] = ele;
		base = (size_t)i * dim_num;
		for (j = 0; j < dim_num; j++)
		{
			inputS >> key_pos >> ch >> ele;
			svm_node_ptr[base + j].index = key_pos;
			svm_node_ptr[base + j].value = ele;
		}
	}

	printf("base=%ld svm_node_ptr[49999].index=%d value=%lf\n", base, svm_node_ptr[base + dim_num - 1].index, svm_node_ptr[base + dim_num - 1].value);

	/*
		r2_read_problem(buf, prob);
		free(buf);
		prob.curSV_num = 0;
		printf("prob.l=%d \n", prob.l );
		error_msg = svm_check_parameter(&prob, &param);
		param.max_iter = 50;
		if (error_msg)
		{
			fprintf(stderr, "ERROR: %s\n", error_msg);
			exit(1);
		}

		if (cross_validation)
		{
			do_cross_validation();
		}
		else
		{
			printf("Begin.. training\n");
			model = svm_train(&prob, &param);
			print_model(model);

			char model_fn[200];
			int cnt = 0;
			for (cnt = 0; cnt < 100; cnt++)
			{
				prob.curSV_num = model->l;
				printf("cnt = %d model->l=%d prob.l=%d", cnt, model->l, prob.l);
				if (prob.curSV_num > 0)
				{
					free(prob.ini_alphas);
					free(prob.ini_indices);
					prob.ini_alphas = Malloc(double, prob.curSV_num);
					prob.ini_indices = Malloc(int, prob.curSV_num);
					for (int i = 0; i < prob.curSV_num; i++)
					{
						prob.ini_alphas[i] = model->sv_coef[0][i] ;
						prob.ini_indices[i] = model->sv_indices[i];
					}
					printf("prob.curSV_num=%d\n", prob.curSV_num );
				}
				model = svm_train(&prob, &param);
				print_model(model);

				sprintf(model_fn, "%s-%d", model_file_name, cnt);
				// get the model parameters
				printf("Saving... %s\n", model_fn );
				if (svm_save_model(model_fn, model))
				{
					fprintf(stderr, "can't save model to file %s\n", model_fn);
					exit(1);
				}

			}
			svm_free_and_destroy_model(&model);

		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		free(line);
	**/

	error_msg = svm_check_parameter(&prob, &param);
	param.max_iter = -1;
	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}
	probs = Malloc(svm_problem, (prob_num + 1) );
	r2_read_problems(buf, probs, prob_num);
	int tid = 0;
	for (tid = 0; tid < prob_num; tid++)
	{
		thread th(sub_svm_train, tid);
		th.detach();
	}

	// main thread
	while (1 == 1)
	{
		this_thread::sleep_for(chrono::seconds(5));
	}


	return 0;
}
void sub_svm_train(int tid)
{
	printf("this is thread %d\n", tid);

	int sub_idx  = tid + 1;
	struct svm_problem* sub_prob = &(probs[sub_idx]);
	sub_prob->curSV_num = 0;
	svm_model* mymodel = svm_train(sub_prob, &param);
	print_model(mymodel);

}
void print_model(svm_model* model)
{
	printf("Print Model Paras\n");
	printf("Class=%d totalSV=%d\n", model->nr_class, model->l );
	printf("-b=%lf\n", *(model->rho) );
	//binary class
	printf("\n");
	printf("SVs l= %d\n", model->l);
	/*
	for (int i = 0; i < model->l; i++)
	{
		printf("%d:%lf\n", model->sv_indices[i], model->sv_coef[0][i] );
	}
	**/
}
void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	svm_cross_validation(&prob, &param, nr_fold, target);
	if (param.svm_type == EPSILON_SVR ||
	        param.svm_type == NU_SVR)
	{
		for (i = 0; i < prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y) * (v - y);
			sumv += v;
			sumy += y;
			sumvv += v * v;
			sumyy += y * y;
			sumvy += v * y;
		}
		printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
		       ((prob.l * sumvy - sumv * sumy) * (prob.l * sumvy - sumv * sumy)) /
		       ((prob.l * sumvv - sumv * sumv) * (prob.l * sumyy - sumy * sumy))
		      );
	}
	else
	{
		for (i = 0; i < prob.l; i++)
			if (target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / prob.l);
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.max_iter = -1;
	cross_validation = 0;

	// parse options
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-') break;
		if (++i >= argc)
			exit_with_help();
		switch (argv[i - 1][1])
		{
		case 's':
			param.svm_type = atoi(argv[i]);
			break;
		case 't':
			param.kernel_type = atoi(argv[i]);
			break;
		case 'd':
			param.degree = atoi(argv[i]);
			break;
		case 'g':
			param.gamma = atof(argv[i]);
			break;
		case 'r':
			param.coef0 = atof(argv[i]);
			break;
		case 'n':
			param.nu = atof(argv[i]);
			break;
		case 'm':
			param.cache_size = atof(argv[i]);
			break;
		case 'c':
			param.C = atof(argv[i]);
			break;
		case 'e':
			param.eps = atof(argv[i]);
			break;
		case 'p':
			param.p = atof(argv[i]);
			break;
		case 'h':
			param.shrinking = atoi(argv[i]);
			break;
		case 'b':
			param.probability = atoi(argv[i]);
			break;
		case 'q':
			print_func = &print_null;
			i--;
			break;
		case 'v':
			cross_validation = 1;
			nr_fold = atoi(argv[i]);
			if (nr_fold < 2)
			{
				fprintf(stderr, "n-fold cross validation: n must >= 2\n");
				exit_with_help();
			}
			break;
		case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label, sizeof(int) * param.nr_weight);
			param.weight = (double *)realloc(param.weight, sizeof(double) * param.nr_weight);
			param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
			param.weight[param.nr_weight - 1] = atof(argv[i]);
			break;
		default:
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
			exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if (i >= argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if (i < argc - 1)
		strcpy(model_file_name, argv[i + 1]);
	else
	{
		char *p = strrchr(argv[i], '/');
		if (p == NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name, "%s.model", p);
	}
}

// read in a problem (in svmlight format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	while (readline(fp) != NULL)
	{
		char *p = strtok(line, " \t"); // label

		// features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements); // elements = dim_num
	printf("elements(dim_num)=%ld\n", elements );

	max_index = 0;
	j = 0;
	for (i = 0; i < prob.l; i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL) // empty line
			exit_input_error(i + 1);

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit_input_error(i + 1);

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}
	printf("j= %ld\n", j);
	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

	if (param.kernel_type == PRECOMPUTED)
		for (i = 0; i < prob.l; i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

//buf=|sample_num(int)|dim_num(int)|label ... label(double)|features...(svm_node)
void r2_read_problem(char*buf, struct svm_problem& myprob)
{
	int* int_ptr = static_cast<int*>(static_cast<void*>(buf));
	int sample_num = int_ptr[0];
	int dim_num = int_ptr[1];
	double* double_ptr = static_cast<double*>(static_cast<void*>((int_ptr + 2)));

	myprob.l = sample_num;

	myprob.y = Malloc(double, myprob.l);
	myprob.x = Malloc(struct svm_node *, myprob.l);

	int i = 0;
	for (i = 0; i < sample_num; i++)
	{
		myprob.y[i] = double_ptr[i];
	}

	struct svm_node* feature_ptr = static_cast<struct svm_node*>(static_cast<void*>((double_ptr + sample_num)));

	int j = 0;
	int max_index = 0;
	int idx = 0;
	for (i = 0 ; i < sample_num; i++)
	{
		myprob.x[i] = Malloc(struct svm_node, dim_num + 1);
		for (j = 0; j < dim_num; j++)
		{
			idx = i * dim_num + j;
			myprob.x[i][j].index = feature_ptr[idx].index;
			myprob.x[i][j].value = feature_ptr[idx].value;
			if (max_index < myprob.x[i][j].index)
			{
				max_index = myprob.x[i][j].index;
			}
		}
		myprob.x[i][j].index = -1;
	}


	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

}

//master prob + sub_problems
void r2_read_problems(char*buf, struct svm_problem* myprobs, int prob_num)
{
	int* int_ptr = static_cast<int*>(static_cast<void*>(buf));
	int sample_num = int_ptr[0];
	int dim_num = int_ptr[1];
	int unit_sample_num = sample_num / prob_num;
	int offset = 0;

	double* double_ptr = static_cast<double*>(static_cast<void*>((int_ptr + 2)));

	svm_problem* master_problem = &(myprobs[0]);
	master_problem->l = sample_num;
	master_problem->y = Malloc(double, master_problem->l);
	master_problem->x = Malloc(struct svm_node *, master_problem->l);

	int i = 0;
	for (i = 0; i < sample_num; i++)
	{
		master_problem->y[i] = double_ptr[i];
	}

	struct svm_node* feature_ptr = static_cast<struct svm_node*>(static_cast<void*>((double_ptr + sample_num)));

	int j = 0;
	int max_index = 0;
	int idx = 0;
	for (i = 0 ; i < sample_num; i++)
	{
		master_problem->x[i] = Malloc(struct svm_node, dim_num + 1);
		for (j = 0; j < dim_num; j++)
		{
			idx = i * dim_num + j;
			master_problem->x[i][j].index = feature_ptr[idx].index;
			master_problem->x[i][j].value = feature_ptr[idx].value;
			if (max_index < master_problem->x[i][j].index)
			{
				max_index = master_problem->x[i][j].index;
			}
		}
		master_problem->x[i][j].index = -1;
	}

	for (i = 1; i <= prob_num; i++)
	{
		svm_problem* slave_problem = &(myprobs[i]);
		slave_problem->l = unit_sample_num;
		if (i == prob_num)
		{
			slave_problem->l = sample_num - (prob_num - 1) * unit_sample_num;
		}
		slave_problem->y = (master_problem->y + offset);
		slave_problem->x = (master_problem->x + offset);
		offset += unit_sample_num;
		//printf("[%d] y0=%lf",i,  );
	}


	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

}