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
#include "KdV.h"

using namespace std;
using namespace cnl;

double cospi(double x) {
	return cos(M_PI * x);
}

timespec getTime() {
	timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	return time;
}

double getInterval(timespec& t1, timespec& t2) {
	return (double)(t2.tv_nsec - t1.tv_nsec)*0.000001 + (double)(t2.tv_sec - t1.tv_sec)*1000.0;
}

void calc(cnl::Processor processor, cnl::KdV2& kdv) {
	kdv.setProcessor(processor);
	kdv.setInitialCondition(cospi);

	timespec stime = getTime();
	kdv.doStep();
	timespec etime = getTime();

	double elapsed = getInterval(stime,etime);
	cout << "# processor = " << kdv.getProcessorName() << endl;
	cout << "# elapsedTime = " << elapsed << " msec" << endl;
	cout << "# u(0) = " << kdv.result.get(0) << endl;
}

int main(int argn, char** argv) {
	int N;
	double dt;
	double T;
	cnl::Processor processor{cnl::Processor::NOTSPECIFIED};

	if (argn == 4) {
		N  = stoi(argv[1]);
		dt = stod(argv[2]);
		T  = stod(argv[3]);
	} else if (argn == 5) {
		if (strcmp(argv[1],"--CPU") == 0) {
			processor = cnl::Processor::CPU;
		} else if (strcmp(argv[1],"--GPU") == 0) {
				processor = cnl::Processor::GPU;
		} else {
			return -1;
		}
		N  = stoi(argv[2]);
		dt = stod(argv[3]);
		T  = stod(argv[4]);
	} else {
		return -1;
	}

	cnl::KdV2 kdv{N, dt, T};
	cout << "# ----------------" << endl;
	cout << "# N  = " << N << endl;
	cout << "# dt = " << dt << endl;
	cout << "# T  = " << T << endl;
	cout << "# steps = " << kdv.getSteps() << endl;

	try {
		if (processor == cnl::Processor::NOTSPECIFIED) {
			calc(cnl::Processor::CPU, kdv);
			calc(cnl::Processor::GPU, kdv);
		} else if (processor == cnl::Processor::CPU) {
			calc(cnl::Processor::CPU, kdv);
		} else if (processor == cnl::Processor::GPU) {
			calc(cnl::Processor::GPU, kdv);
		}
		kdv.save("/tmp/kdv-qt.npy");
	} catch(cnl::Exception& e) {
		cerr << e.message << endl;
	}
	return 0;
}
