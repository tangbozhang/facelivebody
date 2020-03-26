#ifndef _COMMONFUNCTION_H__
#define _COMMONFUNCTION_H__
#include <memory>

#include "MacroHoliday.h"
#include <math.h>

extern "C" {
#include <cblas.h>
}

#ifdef ANDROID_PLATFORM
#undef USING_SIMOD
#endif

#ifdef USING_SIMOD
#include <immintrin.h>
#endif // USING_SIMOD


#include <cmath>

template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
	Dtype* C)
{
	return;
}

template<>
inline void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
}

template<>
inline void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const double alpha, const double* A, const double* B, const double beta,
	double* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
}

#ifdef USING_SIMOD 
inline float simd_dot(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}

template <typename Dtype>
void caffe_cpu_gemm_simod(bool ta, bool tb, 
	const int m, const int n, const int k, const Dtype* A, const Dtype* B, Dtype* C)
{
	return;
}

static void transpose(float *dst, const float *src, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			dst[m * j + i] = src[n * i + j];
		}
	}
}

template <>
inline void caffe_cpu_gemm_simod(bool ta, bool tb, const int m, const int n,
	const int k,const float* A, const float* B, float* C)
{
	std::shared_ptr<float> Bt_data;
	std::shared_ptr<float> At_data;
	if (ta)
	{
		Bt_data.reset(new float[k * m], std::default_delete<float[]>());
		float *At = At_data.get();
		transpose(At, A, k, m);
		A = At;
	}
	if (!tb)
	{
		Bt_data.reset(new float[k * n], std::default_delete<float[]>());
		float *Bt = Bt_data.get();
		transpose(Bt, B, k, n);
		B = Bt;
	}
	const float* x = A;
	for (int i = 0, idx = 0; i < m; ++i) {
		const float* y = B;
		for (int j = 0; j < n; ++j, ++idx) {
			C[idx] = simd_dot(x, y, k);
			y += k;
		}
		x += k;
	}
}
#endif
template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y)
{
	return;
}

template <>
inline void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
	float* y) {
	cblas_scopy(n, x, 1, y, 1);
	cblas_sscal(n, alpha, y, 1);
}

template <>
inline void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
	double* y) {
	cblas_dcopy(n, x, 1, y, 1);
	cblas_dscal(n, alpha, y, 1);
}

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
	const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
	Dtype* y)
{
	return;
}

template <>
inline void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
	const int N, const float alpha, const float* A, const float* x,
	const float beta, float* y) {
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
inline void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
	const int N, const double alpha, const double* A, const double* x,
	const double beta, double* y) {
	cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}


template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y)
{
	return;
}

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
     \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const float b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = std::pow(a[i], b));

#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, double* y) { \
    v##name<double>(n, a, y); \
  }
DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]));
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]));
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]));
template <>
inline void caffe_powx<float>(const int n, const float* a, const float b,
	float* y) {
	vsPowx(n, a, b, y);
}

template <>
inline void caffe_powx<double>(const int n, const double* a, const double b,
	double* y) {
	vdPowx(n, a, b, y);
}


#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);


template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y)
{
	return;
}

template <>
inline void caffe_mul<float>(const int n, const float* a, const float* b,
	float* y) {
	vsMul(n, a, b, y);
}

template <>
inline void caffe_mul<double>(const int n, const double* a, const double* b,
	double* y) {
	vdMul(n, a, b, y);
}

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y)
{
	return;
}

template <>
inline void caffe_div<float>(const int n, const float* a, const float* b,
	float* y) {
	vsDiv(n, a, b, y);
}

template <>
inline void caffe_div<double>(const int n, const double* a, const double* b,
	double* y) {
	vdDiv(n, a, b, y);
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
	if (alpha == 0) {
		memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
		return;
	}
	for (int i = 0; i < N; ++i) {
		Y[i] = alpha;
	}
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

template <>
inline float caffe_cpu_asum<float>(const int n, const float* x) {
	return cblas_sasum(n, x, 1);
}

template <>
inline double caffe_cpu_asum<double>(const int n, const double* x) {
	return cblas_dasum(n, x, 1);
}

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <>
inline void caffe_scal<float>(const int N, const float alpha, float *X) {
	cblas_sscal(N, alpha, X, 1);
}

template <>
inline void caffe_scal<double>(const int N, const double alpha, double *X) {
	cblas_dscal(N, alpha, X, 1);
}

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <>
inline void caffe_sqr<float>(const int n, const float* a, float* y) {
	vsSqr(n, a, y);
}

template <>
inline void caffe_sqr<double>(const int n, const double* a, double* y) {
	vdSqr(n, a, y);
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y)
{
	memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
	Dtype* Y);

template <>
inline void caffe_axpy<float>(const int N, const float alpha, const float* X,
	float* Y) {
	cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
inline void caffe_axpy<double>(const int N, const double alpha, const double* X,
	double* Y) {
	cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <>
inline void caffe_exp<float>(const int n, const float* a, float* y) {
	vsExp(n, a, y);
}

template <>
inline void caffe_exp<double>(const int n, const double* a, double* y) {
	vdExp(n, a, y);
}


#endif
