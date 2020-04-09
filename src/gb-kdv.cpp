<<<<<<< HEAD
/**
   Copyright 2018 HIGASHIMURA Takenori

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include <iostream>
#include <cmath>
#include <getopt.h>
#include <stdlib.h>
#include "KdV.cuh"

using namespace std;
using namespace cnl;

double cospi(double x) {
	return cos(M_PI * x);
}

class Calculator {
private:
	int N{256};
	double dt{1e-5};
	double T{10};
	Processor processor{Processor::CPU};
	KdV2 kdv;
	string opts{"N:d:T:CG"};
	string dataPath;
	string dataDir{"./results"};
public:
	Calculator(){}
	virtual ~Calculator(){}

	bool parseArguments(int argn, char** argv) {
		char opt;
		while ((opt = getopt(argn, argv, opts.c_str())) != EOF) {
			switch(opt)
			{
				case 'N':
					N = atoi(optarg);
					break;
				case 'd':
					dt = atof(optarg);
					break;
				case 'T':
					T = atof(optarg);
					break;
				case 'C':
					processor = Processor::CPU;
					break;
				case 'G':
					processor = Processor::GPU;
					break;
				default:
					return false;
			}
		}
		kdv.setup(N, dt, T);
		kdv.setProcessor(processor);
		ostringstream stream;
		stream << N << '_' << dt << '_' << T;	//<< '_' << kdv.getProcessorName();
		dataPath = dataDir + string("/kdv_") + stream.str() + string(".npy");
		return true;
	}

	bool calc() {
		try {
			kdv.setInitialCondition(cospi);

			timespec stime = getTime();
			kdv.doStep();
			timespec etime = getTime();

			double elapsed = getInterval(stime,etime);
			cout << "# ----------------" << endl;
			cout << "# N  = " << N << endl;
			cout << "# dt = " << dt << endl;
			cout << "# T  = " << T << endl;
			cout << "# steps = " << kdv.getSteps() << endl;
			cout << "# processor = " << kdv.getProcessorName() << endl;
			cout << "# elapsedTime = " << elapsed << " msec" << endl;
			cout << "# u(0) = " << kdv.result.get(0) << endl;

			kdv.save(dataPath);
			return true;
		} catch(cnl::Exception& e) {
			cerr << e.message << endl;
			return false;
		}
	}

private:
=======
/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <cmath>
#include <getopt.h>
#include <stdlib.h>
#include "KdV.h"

using namespace std;
using namespace cnl;

double cospi(double x) {
	return cos(M_PI * x);
}

class Calculator {
private:
	int N{256};
	double dt{1e-5};
	double T{10};
	bool both{false};
	Processor processor{Processor::CPU};
	KdV2 kdv;
	const char* opts{"N:d:T:CO:G"};
public:
	Calculator(){}
	virtual ~Calculator(){}

	bool parseArguments(int argn, char** argv) {
		char opt;
		while ((opt = getopt(argn, argv, opts)) != EOF) {
			switch(opt)
			{
				case 'N':
					N = atoi(optarg);
					break;
				case 'd':
					dt = atof(optarg);
					break;
				case 'T':
					T = atof(optarg);
					break;
				case 'C':
					both = true;
					break;
				case 'G':
					processor = Processor::GPU;
					break;
				default:
					return false;
			}
		}
		return true;
	}

	bool calc() {
		kdv.setup(N, dt, T);
		try {
			if (both) {
				processor = Processor::CPU;
				calcOnProcessor();
				processor = Processor::GPU;
				calcOnProcessor();
			} else {
				calcOnProcessor();
			}
			kdv.save("./results/kdv-qt.npy");
			return true;
		} catch(cnl::Exception& e) {
			cerr << e.message << endl;
			return false;
		}
	}

private:
	void calcOnProcessor() {
		kdv.setProcessor(processor);
		kdv.setInitialCondition(cospi);

		timespec stime = getTime();
		kdv.doStep();
		timespec etime = getTime();

		double elapsed = getInterval(stime,etime);
		cout << "# ----------------" << endl;
		cout << "# N  = " << N << endl;
		cout << "# dt = " << dt << endl;
		cout << "# T  = " << T << endl;
		cout << "# steps = " << kdv.getSteps() << endl;
		cout << "# processor = " << kdv.getProcessorName() << endl;
		cout << "# elapsedTime = " << elapsed << " msec" << endl;
		cout << "# u(0) = " << kdv.result.get(0) << endl;
	}

>>>>>>> refs/remotes/eclipse_auto/master
	timespec getTime() {
		timespec time;
		clock_gettime(CLOCK_MONOTONIC, &time);
		return time;
	}

	double getInterval(timespec& t1, timespec& t2) {
		return (double)(t2.tv_nsec - t1.tv_nsec)*0.000001 + (double)(t2.tv_sec - t1.tv_sec)*1000.0;
	}
};

int main(int argn, char** argv) {
	Calculator calculator;

	if (calculator.parseArguments(argn, argv) == false) {
		return -1;
	}
	if (calculator.calc() == false) {
		return -1;;
	}
	return 0;
}
