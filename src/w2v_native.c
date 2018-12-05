/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_get_vocab(SEXP train_file, SEXP output_file, SEXP read_vocab_file, SEXP verbose_);
extern SEXP R_w2v(SEXP train_file, SEXP output_file, SEXP read_vocab_file,
  SEXP binary, SEXP disk, SEXP negative, SEXP iter, SEXP window, SEXP batch_size, SEXP min_count, SEXP hidden_size, SEXP min_sync_words, SEXP full_sync_times, SEXP alpha, SEXP sample, SEXP model_sync_period,
  SEXP nthreads, SEXP message_size, SEXP verbose_);

static const R_CallMethodDef CallEntries[] = {
  {"R_get_vocab", (DL_FUNC) &R_get_vocab, 3},
  {"R_w2v", (DL_FUNC) &R_w2v, 19},
  {NULL, NULL, 0}
};

void R_init_w2v(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
