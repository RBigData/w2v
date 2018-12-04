#include "pWord2Vec/pWord2Vec.h"
#include "pWord2Vec/types.h"
#include <Rinternals.h>

#define CHARPT(x,i) ((char*)CHAR(STRING_ELT(x,i)))


extern "C" SEXP R_get_vocab(SEXP train_file, SEXP vocab_file, SEXP verbose_)
{
  file_params_t files;
  
  files.train_file = CHARPT(train_file, 0);
  files.output_file = NULL;
  files.save_vocab_file = CHARPT(vocab_file, 0);
  files.read_vocab_file = NULL;
  
  bool verbose = (bool) INTEGER(verbose_)[0];
  
  get_vocab(&files, verbose);
  
  return R_NilValue;
}
