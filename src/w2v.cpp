#include "pWord2Vec/pWord2Vec.h"
#include <Rinternals.h>


#define CHARPT(x,i) ((char*)CHAR(STRING_ELT(x,i)))

// printf("parallel word2vec (sgns) in distributed memory system\n\n");
// printf("Options:\n");
// printf("Parameters for training:\n");
// printf("\t-train <file>\n");
// printf("\t\tUse text data from <file> to train the model\n");
// printf("\t-output <file>\n");
// printf("\t\tUse <file> to save the resulting word vectors\n");
// printf("\t-size <int>\n");
// printf("\t\tSet size of word vectors; default is 100\n");
// printf("\t-window <int>\n");
// printf("\t\tSet max skip length between words; default is 5\n");
// printf("\t-sample <float>\n");
// printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
// printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
// printf("\t-negative <int>\n");
// printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
// printf("\t-threads <int>\n");
// printf("\t\tUse <int> threads (default 12)\n");
// printf("\t-iter <int>\n");
// printf("\t\tNumber of training iterations; default is 5\n");
// printf("\t-min-count <int>\n");
// printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
// printf("\t-alpha <float>\n");
// printf("\t\tSet the starting learning rate; default is 0.1\n");
// printf("\t-debug <int>\n");
// printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
// printf("\t-binary <int>\n");
// printf("\t\tSave the resulting vectors in binary mode; default is 0 (off)\n");
// printf("\t-save-vocab <file>\n");
// printf("\t\tThe vocabulary will be saved to <file>\n");
// printf("\t-read-vocab <file>\n");
// printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
// printf("\t-batch-size <int>\n");
// printf("\t\tThe batch size used for mini-batch training; default is 11 (i.e., 2 * window + 1)\n");
// printf("\t-disk\n");
// printf("\t\tStream text from disk during training, otherwise the text will be loaded into memory before training\n");
// printf("\t-sync-period <float>\n");
// printf("\t\tSynchronize model every <float> seconds; default is 0.1\n");
// printf("\t-min-sync-words <int>\n");
// printf("\t\tMinimal number of words to be synced at each model sync; default is 1024\n");
// printf("\t-full-sync-times <int>\n");
// printf("\t\tEnforced full model sync-up times during training; default is 0\n");
// printf("\t-message-size <int>\n");
// printf("\t\tMPI message chunk size in MB; default is 1024MB\n");
// printf("\nExamples:\n");
// printf("mpirun -np 1 ./word2vec_mpi -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");

extern "C" SEXP R_w2v(SEXP train, SEXP verbose_)
{
  w2v_params_t p;
  
  p.verbose = (bool) INTEGER(verbose_)[0];
  p.binary = 0;
  p.disk = false;
  p.num_threads = 4;
  p.negative = 5;
  p.iter = 5;
  p.window = 5;
  p.batch_size = 11;
  p.min_count = 5;
  p.min_reduce = 1;
  p.vocab_max_size = 1000;
  p.vocab_size = 0;
  p.hidden_size = 100;
  p.min_sync_words = 1024;
  p.full_sync_times = 0;
  p.message_size = 1024;
  p.train_words = 0;
  p.alpha = 0.1f;
  p.sample = 1e-3f;
  p.model_sync_period = 0.1f;
  
  w2v(&p, CHARPT(train, 0));
  
  return R_NilValue;
}
