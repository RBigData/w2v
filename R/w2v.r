#' w2v
#' 
#' Bindings for MPI+threads implementation of word2vec.
#' 
#' @param params
#' A parameters list. See \code{?w2v_params}.
#' @param train_file
#' Input plaintext file.
#' @param output_file
#' TODO
#' @param vocab_file
#' Either a file path (string) pointing to the output of \code{get_vocab()} if
#' the vocabulary has been pre-computed, or \code{NULL} to compute it on the
#' fly.
#' @param nthreads
#' TODO
#' @param message_size
#' MPI message chunk size in MB.
#' @param verbose
#' Want it to print what it's doing?
#' 
#' @return
#' Invisibly returns \code{NULL}.
#' 
#' @useDynLib w2v R_w2v
#' @export
w2v = function(params=w2v_params(), train_file, output_file=NULL, vocab_file=NULL,
  nthreads=4, message_size=1024, verbose=FALSE)
{
  train_file = path.expand(train_file)
  if (!file.exists(train_file))
    comm.stop("train_file does not exist")
  if (!is.null(output_file))
    output_file = path.expand(output_file)
  if (!is.null(vocab_file))
  {
    read_vocab_file = path.expand(vocab_file)
    if (!file.exists(vocab_file))
      comm.stop("vocab_file does not exist")
  }
  
  .Call(R_w2v, train_file, output_file, vocab_file,
    params$binary, params$disk, params$negative, params$iter, params$window, params$batch_size, params$min_count, params$hidden_size, params$min_sync_words, params$full_sync_times, params$alpha, params$sample, params$model_sync_period,
    as.integer(nthreads), as.integer(message_size), as.logical(verbose)
  )
  
  invisible()
}
