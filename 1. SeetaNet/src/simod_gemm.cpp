#include "simod_gemm.h"

#ifdef USING_SIMOD


#include <immintrin.h>

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

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) 
    {
        for (k = 0; k < K; ++k) 
        {
            register float A_PART = ALPHA*A[i*lda + k];
            j = 0;

            __m128  X,Y;
            X = _mm_set_ps(A_PART, A_PART, A_PART, A_PART);
            
            for (; j < N-3; j += 4) 
            {
                float* b_pos = &B[k*ldb + j];
                Y = _mm_loadu_ps(b_pos);
                __m128 acc = _mm_mul_ps(X, Y);

                float* current = &C[i*ldc + j];
                __m128 c_base = _mm_loadu_ps(current);
                
                acc = _mm_add_ps(acc, c_base);                
                _mm_storeu_ps(current, acc);
            }
            for (j; j < N; ++j) 
            {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    if (1 == ALPHA)
    {
        const float* x = A;
        for (int i = 0, idx = 0; i < M; ++i) {
            const float* y = B;
            for (int j = 0; j < N; ++j, ++idx) {
                C[idx] = simd_dot(x, y, K);
                y += K;
            }
            x += K;
        }
    }
    else
    {
        int i, j, k;
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                register float sum = 0;
                for (k = 0; k < K; ++k) {
                    sum += ALPHA*A[i*lda + k] * B[j*ldb + k];
                }
                C[i*ldc + j] += sum;
            }
        }
    }
    
}

void gemm_tn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register float A_PART = ALPHA*A[k*lda + i];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            register float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA*A[i + k*lda] * B[k + j*ldb];
            }
            C[i*ldc + j] += sum;
        }
    }
}


template<>
void gemm_cpu(bool TA, bool TB, int M, int N, int K, float ALPHA,
    float *A,
    float *B,
    float BETA,
    float *C)
{
    int lda = (TA == false) ? K : M;
    int ldb = (TB == false) ? N : K;

    int ldc = N;
    int i = 0;

    __m128 X = _mm_set_ps(BETA, BETA, BETA, BETA);

    for (i=0;i < M * N; i+=4 )
    {
        __m128 Y = _mm_loadu_ps(C+i);
        __m128 result = _mm_mul_ps(X, Y);
        _mm_storeu_ps((C+i), result);
    }

    for (; i < M * N; i++)
    {
        C[i] *= BETA;
    }
   
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

template<>
void gemm_cpu(bool TA, bool TB, int M, int N, int K, double ALPHA,
    double *A,
    double *B,
    double BETA,
    double *C)
{

}


#endif // !ANDROID_CPU