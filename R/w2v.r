#' w2v
#' 
#' TODO
#' 
#' @details
#' TODO
#' 
#' @param train_file
#' 
#' @param output_file
#' 
#' @param vocab_file
#' If \code{NULL}, 
#' @param nthreads
#' 
#' @param message_size
#' 
#' @param verbose
#' 
#' 
#' @return
#' Invisibly returns \code{NULL}.
#' 
#' @useDynLib w2v R_w2v
#' @export
w2v = function(train_file, output_file=NULL, vocab_file=NULL,
  nthreads=4, message_size=1024, verbose=FALSE)
{
  train_file = path.expand(train_file)
  if (!file.exists(train_file))
    comm.stop("train_file does not exist")
  if (!is.null(output_file))
    output_file = path.expand(output_file)
  if (!is.null(vocab_file))
    read_vocab_file = path.expand(vocab_file)
  
  .Call(R_w2v, train_file, output_file, vocab_file,
    as.integer(nthreads), as.integer(message_size), as.logical(verbose)
  )
  
  invisible()
}
