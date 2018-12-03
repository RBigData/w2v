/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_w2v(SEXP train, SEXP verbose_);

static const R_CallMethodDef CallEntries[] = {
  {"R_w2v", (DL_FUNC) &R_w2v, 2},
  {NULL, NULL, 0}
};

void R_init_w2v(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
