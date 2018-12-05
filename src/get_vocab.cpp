#include "pWord2Vec/pWord2Vec.h"
#include "pWord2Vec/types.h"
#include "Rw2v.h"


extern "C" SEXP R_get_vocab(SEXP train_file, SEXP vocab_file, SEXP comm_ptr, SEXP verbose)
{
  sys_params_t sys;
  file_params_t files;
  
  sys.comm = *(get_mpi_comm_from_Robj(comm_ptr));
  sys.verbose = (bool) INTEGER(verbose)[0];
  
  files.train_file = CHARPT(train_file, 0);
  files.output_file = NULL;
  files.save_vocab_file = CHARPT(vocab_file, 0);
  files.read_vocab_file = NULL;
  
  get_vocab(&sys, &files);
  
  return R_NilValue;
}
