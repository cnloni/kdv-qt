/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "KdV.h"

__inline__ __device__ double derivative(unsigned tid, int N, cnl::KdVParam kdvp, double* a) {
	double p1 = a[(tid + 1)%N];
	double m1 = a[(tid - 1 + N)%N];
	double p2 = a[(tid + 2)%N];
	double m2 = a[(tid - 2 + N)%N];
	return - kdvp.a * (p1 + a[tid] + m1) * (p1 - m1)
			- kdvp.b * ((p2 - m2) - 2*(p1 - m1));
}

/**
 * dt^2の精度
 */
__global__ void kdv2_step(long nSteps, int N, cnl::KdVParam kdvp, double* um1) {
	unsigned tid = threadIdx.x;

	double f;

	// 共有メモリを割付
	extern __shared__ double a[];
	double s;
	double sm1 = um1[tid];

	{
		//第1ステップ
		a[tid] = sm1;
		__syncthreads();

		f = derivative(tid, N, kdvp, a);
		__syncthreads();

		a[tid] = s = sm1 + f;
		__syncthreads();

		f = derivative(tid, N, kdvp, a);
		__syncthreads();

		nSteps--;
	}

	for(int i=0; i<nSteps; i++) {
		double sb = sm1;
		sm1 = s;
		a[tid] = s = sb + 2*f;
		__syncthreads();

		f = derivative(tid, N, kdvp, a);
		__syncthreads();
	}
	um1[tid] = s ;
}

__global__ void kdv3_step(long nSteps, int N, cnl::KdVParam kdvp, double* um1) {
	unsigned tid = threadIdx.x;

	double fm1, fm2, f;

	// 共有メモリを割付
	extern __shared__ double a[];
	double s;
	double sm1 = um1[tid];

	{
		//第1ステップ
		a[tid] = sm1;
		__syncthreads();

		fm1 = derivative(tid, N, kdvp, a);
		__syncthreads();

		a[tid] = s = sm1 + fm1;
		__syncthreads();

		f = derivative(tid, N, kdvp, a);
		__syncthreads();

		a[tid] = s += (f - fm1) / 2;
		__syncthreads();

		f = derivative(tid, N, kdvp, a);
		__syncthreads();
	}

	for(int i=1; i<nSteps; i++) {
		double sb = sm1;
		sm1 = s;
		a[tid] = s += 2*f;
		__syncthreads();

		fm2 = fm1;
		fm1 = f;
		f = derivative(tid, N, kdvp, a);
		__syncthreads();

		a[tid] = s = sm1 + (-fm2 + 8*fm1 + 5*f) / 12;
		__syncthreads();

		f = derivative(tid, N, kdvp, a);
		__syncthreads();
	}
	um1[tid] = s ;
}
