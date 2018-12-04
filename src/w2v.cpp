#include "pWord2Vec/pWord2Vec.h"
#include <Rinternals.h>

#define CHARPT(x,i) ((char*)CHAR(STRING_ELT(x,i)))


extern "C" SEXP R_w2v(SEXP train, SEXP verbose_)
{
  w2v_params_t p;
  sys_params_t sys;
  file_params_t files;
  
  p.binary = false;
  p.disk = false;
  p.negative = 5;
  p.iter = 5;
  p.window = 5;
  p.batch_size = 11;
  p.min_count = 5;
  p.hidden_size = 100;
  p.min_sync_words = 1024;
  p.full_sync_times = 0;
  p.alpha = 0.1f;
  p.sample = 1e-3f;
  p.model_sync_period = 0.1f;
  
  sys.verbose = (bool) INTEGER(verbose_)[0];
  sys.num_threads = 4;
  sys.message_size = 1024;
  
  files.train_file = CHARPT(train, 0);
  files.output_file = "output.txt";
  files.save_vocab_file = "save_vocab_file.txt";
  files.read_vocab_file = 0;
  
  w2v(&p, &sys, &files);
  
  return R_NilValue;
}
