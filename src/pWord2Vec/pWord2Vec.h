#ifndef PWORD2VEC_H_
#define PWORD2VEC_H_


#include <cstdio>
#include <cstdlib>


// TODO check for icc first
#define _mm_malloc(a,b) malloc(a)
#define _mm_free free



#define MAX_STRING 256
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MPI_SCALAR MPI_FLOAT

typedef float real;
typedef unsigned int uint;
typedef unsigned long long ulonglong;



typedef struct
{
  int binary;
  bool verbose;
  bool disk;
  int num_threads;
  int negative;
  int iter;
  int window;
  int batch_size;
  unsigned int min_count;
  unsigned int min_reduce;
  int vocab_max_size;
  int vocab_size;
  int hidden_size;
  int min_sync_words;
  int full_sync_times;
  int message_size; // MB
  ulonglong train_words;
  real alpha;
  real sample;
  real model_sync_period;
} w2v_params_t;



void w2v(w2v_params_t *p, char *train);


#endif
