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
  bool binary;
  // Save the resulting vectors in binary mode?
  bool disk;
  // Stream text from disk during training? Otherwise the text will be loaded into memory before training
  int negative;
  // Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
  int iter;
  // Number of training iterations
  int window;
  // Set max skip length between words
  int batch_size;
  // The batch size used for mini-batch training
  uint min_count;
  // This will discard words that appear less than <int> times
  int hidden_size;
  // size of word vectors
  int min_sync_words;
  // Minimal number of words to be synced at each model sync
  int full_sync_times;
  // Enforced full model sync-up times during training
  real alpha;
  // the starting learning rate
  real sample;
  // threshold for occurrence of words. Those that appear with higher frequency in the training data
  real model_sync_period;
  // Synchronize model every <float> seconds; default is 0.1
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
