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

#include "KdV.h"

/**
 * GPUから呼ばれる(__device__)インライン(__inline__)関数
 */
__inline__ __device__ double derivative(unsigned tid, int N, cnl::KdVParam kdvp, double* a) {
	double p1 = a[(tid + 1)%N];
	double m1 = a[(tid - 1 + N)%N];
	double p2 = a[(tid + 2)%N];
	double m2 = a[(tid - 2 + N)%N];
	return - kdvp.a * (p1 + a[tid] + m1) * (p1 - m1)
			- kdvp.b * ((p2 - m2) - 2*(p1 - m1));
}

/**
 * CPUから呼ばれる(__global__)GPU上の関数
 */
__global__ void kdv2_step(long nSteps, int N, cnl::KdVParam kdvp, double* um1) {
	unsigned tid = threadIdx.x;

	double f;

	// 共有メモリを割付
	extern __shared__ double a[];
	double s;
	double sm1 = um1[tid];

	//第1ステップ
	//共有メモリに代入する。それぞれのスレッドが自分の担当する位置に振幅の値を代入するので、
	//結果的に共有メモリの全エリアに代入が行われる。
	//そのため、__suncthreads()で代入が完了するのを待っている
	a[tid] = sm1;
	__syncthreads();

	//2軒両隣までの共有メモリを参照して、方程式の右辺を計算する。
	//後に共有メモリへの代入があるため、__suncthreads()ですべての参照が完了するのを待つ
	f = derivative(tid, N, kdvp, a);
	__syncthreads();

	a[tid] = s = sm1 + f;
	__syncthreads();

	f = derivative(tid, N, kdvp, a);
	__syncthreads();

	nSteps--;

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
