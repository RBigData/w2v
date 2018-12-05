#' get_vocab
#' 
#' Pre-compute the vocabulary for use with \code{w2v()}.
#' 
#' @param train_file
#' Input plaintext file.
#' @param vocab_file
#' File path (string) pointing to where you want the vocabulary to be saved.
#' @param verbose
#' Want it to print what it's doing?
#' 
#' @return
#' Invisibly returns \code{NULL}.
#' 
#' @useDynLib w2v R_get_vocab
#' @export
get_vocab = function(train_file, vocab_file, comm=0, verbose=FALSE)
{
  check.is.string(train_file)
  check.is.string(vocab_file)
  check.is.flag(verbose)
  
  train_file = path.expand(train_file)
  if (!file.exists(train_file))
    comm.stop("train_file does not exist")
  vocab_file = path.expand(vocab_file)
  
  comm_ptr = pbdMPI::get.mpi.comm.ptr(comm)
  
  .Call(R_get_vocab, train_file, vocab_file, comm_ptr, verbose)
  invisible()
}
