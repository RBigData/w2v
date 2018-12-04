#include "pWord2Vec/pWord2Vec.h"
#include "pWord2Vec/types.h"
#include <Rinternals.h>

#define CHARPT(x,i) ((char*)CHAR(STRING_ELT(x,i)))


extern "C" SEXP R_w2v(SEXP train_file, SEXP output_file, SEXP read_vocab_file,
  SEXP nthreads, SEXP message_size, SEXP verbose_)
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
  p.alpha = 0.1;
  p.sample = 1e-3;
  p.model_sync_period = 0.1;
  
  sys.num_threads = INTEGER(nthreads)[0];
  sys.message_size = INTEGER(message_size)[0];
  sys.verbose = (bool) INTEGER(verbose_)[0];
  
  files.train_file = CHARPT(train_file, 0);
  files.output_file = isNull(output_file) ? NULL : CHARPT(output_file, 0);
  files.save_vocab_file = NULL;
  files.read_vocab_file = isNull(read_vocab_file) ? NULL : CHARPT(read_vocab_file, 0);
  
  w2v(&p, &sys, &files);
  
  return R_NilValue;
}
