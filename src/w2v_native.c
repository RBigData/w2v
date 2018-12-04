/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_get_vocab(SEXP train_file, SEXP output_file, SEXP read_vocab_file, SEXP verbose_);
extern SEXP R_w2v(SEXP train, SEXP verbose_);

static const R_CallMethodDef CallEntries[] = {
  {"R_get_vocab", (DL_FUNC) &R_get_vocab, 3},
  {"R_w2v", (DL_FUNC) &R_w2v, 4},
  {NULL, NULL, 0}
};

void R_init_w2v(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
