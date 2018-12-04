#' get_vocab
#' 
#' TODO
#' 
#' @details
#' TODO
#' 
#' @param train_file
#' TODO
#' @param vocab_file
#' TODO
#' 
#' @useDynLib w2v R_get_vocab
#' @export
get_vocab = function(train_file, vocab_file, verbose=FALSE)
{
  train_file = path.expand(train_file)
  if (!file.exists(train_file))
    comm.stop("train_file does not exist")
  vocab_file = path.expand(vocab_file)
  
  .Call(R_get_vocab, train_file, vocab_file, verbose)
  invisible()
}
