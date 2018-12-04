// Copyright (C) 2018 ORNL

#ifndef PWORD2VEC_BLAS_H_
#define PWORD2VEC_BLAS_H_

#include "blas_backend.h"

#define USE_CBLAS

#ifdef BLAS_USE_MKL
  void mkl_set_num_threads(int nt);
  mkl_set_num_threads(1);
#elif BLAS_USE_OPENBLAS
  void openblas_set_num_threads(int num_threads);
  openblas_set_num_threads(1);
#endif


#endif