#include "pWord2Vec/pWord2Vec.h"
#include "pWord2Vec/types.h"
#include "Rw2v.h"


extern "C" SEXP R_w2v(SEXP train_file, SEXP output_file, SEXP read_vocab_file,
  SEXP binary, SEXP disk, SEXP negative, SEXP iter, SEXP window, SEXP batch_size, SEXP min_count, SEXP hidden_size, SEXP min_sync_words, SEXP full_sync_times, SEXP alpha, SEXP sample, SEXP model_sync_period,
  SEXP nthreads, SEXP message_size, SEXP comm_ptr, SEXP verbose_)
{
  w2v_params_t p;
  sys_params_t sys;
  file_params_t files;
  
  p.binary = (bool) LOGICAL(binary)[0];
  p.disk = (bool) LOGICAL(disk)[0];
  p.negative = INTEGER(negative)[0];
  p.iter = INTEGER(iter)[0];
  p.window = INTEGER(window)[0];
  p.batch_size = INTEGER(batch_size)[0];
  p.min_count = INTEGER(min_count)[0];
  p.hidden_size = INTEGER(hidden_size)[0];
  p.min_sync_words = INTEGER(min_sync_words)[0];
  p.full_sync_times = INTEGER(full_sync_times)[0];
  p.alpha = (real) REAL(alpha)[0];
  p.sample = (real) REAL(sample)[0];
  p.model_sync_period = (real) REAL(model_sync_period)[0];
  
  sys.num_threads = INTEGER(nthreads)[0];
  sys.message_size = INTEGER(message_size)[0];
  sys.comm = *(get_mpi_comm_from_Robj(comm_ptr));
  sys.verbose = (bool) INTEGER(verbose_)[0];
  
  files.train_file = CHARPT(train_file, 0);
  files.output_file = isNull(output_file) ? NULL : CHARPT(output_file, 0);
  files.save_vocab_file = NULL;
  files.read_vocab_file = isNull(read_vocab_file) ? NULL : CHARPT(read_vocab_file, 0);
  
  w2v(&p, &sys, &files);
  
  return R_NilValue;
}
