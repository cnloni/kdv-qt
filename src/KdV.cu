/*
 * KdV.cpp
 *
 *  Created on: 2017/07/20
 *      Author: oni
 */

#include <helper_cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "KdV.h"

__global__ void kdv2_step(long nSteps, int N, cnl::KdVParam kdvp, double* u1);

namespace cnl {

/**
 * 初期条件を、従属変数u[j]にセットする
 * N: 位置方向の計算点の数
 * fn: 初期状態を与える関数
 */
void KdV2::setInitialCondition(double (*fn)(double)) {
	double x;
	for(int j=0; j<N; j++) {
		x = (double)j * dx;
		u[j] = (*fn)(x);
	}
	counter = 0;
}

/**
 * KdV方程式の右辺f_i^jを計算する
 * N: 位置方向の計算点の数
 * uu: 従属変数（位置方向の数列）、N個の配列
 * ff: 右辺（位置方向の数列）、N個の配列
 * kdvp: 係数を格納するクラス
 */
void KdV2::calculateDerivatives(double *uu, double *ff) {
	for(int j=0; j<N; j++) {
		double p1 = uu[(j + 1)%N];
		double m1 = uu[(j - 1 + N)%N];
		double p2 = uu[(j + 2)%N];
		double m2 = uu[(j - 2 + N)%N];
		ff[j] = - kdvp.a * (p1 + uu[j] + m1) * (p1 - m1)
			- kdvp.b * ((p2 - m2) - 2*(p1 - m1));
	}
}

/**
 * CPUを使う計算
 * N: 位置方向の計算点の数
 * u[j]: u_i^j、N個の配列
 * um1[j]: u_{i-1}^j、N個の配列
 * counter: 時間方向の計算度数
 * nSteps: 計算度数の上限
 * result: 結果を格納するクラス
 */
void KdV2::doStepOnCPU() {
	if (counter == 0) {
		calculateDerivatives(u, f);
		double* ub = um1;
		um1 = u;
		u = ub;
		for(int j=0; j<N; j++) {
			u[j] = um1[j] + f[j];
		}
		calculateDerivatives(u, f);
		counter = 1;
	}
	for(; counter<nSteps; counter++) {
		double* ub = um1;
		um1 = u;
		u = ub;

		for(int j=0; j<N; j++) {
			u[j] += 2*f[j];
		}
		calculateDerivatives(u, f);
	}
	result.setAll(u);
}

/**
 * GPUを使う計算
 * N: 位置方向の計算点の数
 * u[j]: u_i^j、N個の配列
 * gu[j]: u[j]に対応する、GPU上のグローバル配列
 * counter: 時間方向の計算度数
 * nSteps: 計算度数の上限
 * result: 結果を格納するクラス
 */
void KdV2::doStepOnGPU() {
	size_t unit = N * sizeof(double);

	//GPU上のグローバルメモリを、double✕N個分確保する
	double* gu;
	checkCudaErrors(cudaMalloc((void**)&gu, unit));

	//初期状態を入れた配列u[i]を、GPU上のメモリにコピーする
	checkCudaErrors(cudaMemcpy(gu, u, unit, cudaMemcpyHostToDevice));

	//GPUのグローバル関数
	//グリッドサイズ=1
	//ブロックサイズ=N
	//共有メモリサイズ=double✕N
	kdv2_step<<<1, N, unit>>>(nSteps, N, kdvp, gu);

	//カーネルプログラムの実行終了を待つ（この場合は、後のcudaMemcpy()でブロッキングされるので、無くても良い）
	cudaDeviceSynchronize();

	//計算結果（GPUのグローバルメモリ）をシステムメモリにコピーする
	checkCudaErrors(cudaMemcpy(result.head(), gu, unit, cudaMemcpyDeviceToHost));

	//GPUメモリを開放する
	checkCudaErrors(cudaFree(gu));
	counter += nSteps;
}

} /* namespace cnl */
