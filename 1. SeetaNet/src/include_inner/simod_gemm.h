#include "MacroHoliday.h"
#ifdef USING_SIMOD
#ifndef __SIMOD_GEMM_H_
#define __SIMOD_GEMM_H_



template<class T>
void gemm_cpu(bool TA, bool TB, int M, int N, int K, T ALPHA,
    T *A,
    T *B,
    T BETA,
    T *C);




#endif // !__SIMOD_GEMM_H_
#endif 