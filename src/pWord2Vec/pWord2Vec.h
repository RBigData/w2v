#ifndef PWORD2VEC_H_
#define PWORD2VEC_H_


#include <cstdio>
#include <cstdlib>

#include "types.h"


#ifndef __INTEL_COMPILER
  // #define _mm_malloc(a,b) malloc(a)
  static inline void* _mm_malloc(size_t size, size_t alignment)
  {
    return aligned_alloc(alignment, size);
  }
  #define _mm_free free
#endif


#define MAX_STRING 64
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MPI_SCALAR MPI_FLOAT


typedef struct
{
  // Save the resulting vectors in binary mode?
  bool binary;
  // Stream text from disk during training? Otherwise the text will be loaded into memory before training
  bool disk;
  // Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
  int negative;
  // Number of training iterations
  int iter;
  // Set max skip length between words
  int window;
  // The batch size used for mini-batch training
  int batch_size;
  // This will discard words that appear less than <int> times
  uint min_count;
  // size of word vectors
  int hidden_size;
  // Minimal number of words to be synced at each model sync
  int min_sync_words;
  // Enforced full model sync-up times during training
  int full_sync_times;
  // the starting learning rate
  real alpha;
  // threshold for occurrence of words. Those that appear with higher frequency in the training data
  real sample;
  // Synchronize model every <float> seconds; default is 0.1
  real model_sync_period;
} w2v_params_t;



typedef struct
{
  int num_threads;
  // MPI message chunk size in MB
  int message_size;
  bool verbose;
} sys_params_t;



typedef struct
{
  // input data
  char *train_file;
  // output word vectors
  char *output_file;
  // vocabulary will be saved
  char *save_vocab_file;
  // The vocabulary will be read from <file>, not constructed from the training data
  char *read_vocab_file;
} file_params_t;


void get_vocab(file_params_t *files, bool verbose_);
void w2v(w2v_params_t *p, sys_params_t *sys, file_params_t *files);


#endif
