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

#ifndef KDV_H_
#define KDV_H_

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>
#include "CnlArray.h"

using namespace std;

namespace cnl {

enum class Processor {
	NOTSPECIFIED = 0,
	CPU,
	GPU,
};

struct KdVParam {
	double a, b;
};

class KdV2 {
public:
	KdVParam kdvp;
	NumpyArray result;
protected:
	Processor processor{Processor::CPU};
	double delta = 0.022;
	double TB = 1 / M_PI;
	int N{0};
	double T{0};
	double *u{nullptr};
	double *um1{nullptr};
	double *f{nullptr};
	double dt, dx;
	long nSteps{0};
	long counter{0};
public:
	KdV2(){}
	KdV2(int N, double dt, double T){
		setup(N, dt, T);
	}
	virtual ~KdV2(){
		if (u != nullptr) {
			delete []u;
			u = nullptr;
		}
		if (um1 != nullptr) {
			delete []um1;
			um1 = nullptr;
		}
		if (f != nullptr) {
			delete []f;
			f = nullptr;
		}
	}
	void setup(int N, double dt, double T) {
		this->N = N;
		this->dt = dt;
		this->T = T;
		dx = 2/(double)N;
		nSteps = round(T / dt);
		kdvp.a = dt * TB / dx / 6;
		kdvp.b = delta * delta * dt * TB / (dx * dx * dx) / 2;

		result.setExtension(N);
		u = new double[N];
		um1 = new double[N];
	}
	inline string getProcessorName() {
		if (processor == Processor::GPU) {
			return "GPU";
		} else {
			return "CPU";
		}
	}
	inline void setProcessor(Processor p) {
		processor = p;
	}
	inline double get(int i) {
		if (i >= 0 && i < N) {
			return u[i];
		} else {
			return numeric_limits<double>::quiet_NaN();
		}
	}
	inline double getTime() {
		return (double)counter * dt;
	}
	inline long getCount() {
		return counter;
	}
	inline long getSteps() {
		return nSteps;
	}
	inline void save(string outputFilename) {
		result.open(outputFilename);
		result.save();
		result.close();
	}
	inline void dump() {
		result.dump();
	}
	virtual inline void doStep() {
		cout << "# class = KdV2" << endl;
		if (processor == Processor::GPU) {
			doStepOnGPU();
		} else {
			if (f == nullptr) {
				f = new double[N];
			}
			doStepOnCPU();
		}
	}
	void setInitialCondition(double (*fn)(double));
	void calculateDerivatives(double *uu, double *ff);
	virtual void doStepOnCPU();
	virtual void doStepOnGPU();
};

} /* namespace cnl */

#endif /* KDV_H_ */
