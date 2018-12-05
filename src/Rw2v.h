#ifndef RW2V_H_
#define RW2V_H_


#include <mpi.h>
#include <Rinternals.h>

#define CHARPT(x,i) ((char*)CHAR(STRING_ELT(x,i)))

static inline MPI_Comm* get_mpi_comm_from_Robj(SEXP comm_)
{
  MPI_Comm *comm = (MPI_Comm*) R_ExternalPtrAddr(comm_);
  return comm;
}


#endif
