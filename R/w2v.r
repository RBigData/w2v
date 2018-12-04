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
#' If a string is provided, it will either write 
#' If \code{NULL}, neither saving nor reading takes place.
#' 
#' @useDynLib w2v R_w2v
#' @export
w2v = function(train_file, output_file, vocab_file=NULL, verbose=FALSE)
{
  train_file = path.expand(train_file)
  if (!file.exists(train_file))
    comm.stop("train_file does not exist")
  output_file = path.expand(output_file)
  
  save_vocab_file = NULL
  read_vocab_file = NULL
  if (!is.null(vocab_file))
  {
    vocab_file = path.expand(vocab_file)
    if (file.exists(vocab_file))
      read_vocab_file = vocab_file
    else
      save_vocab_file = vocab_file
  }
  
  .Call(R_w2v,
    train_file, output_file, save_vocab_file, read_vocab_file,
    verbose)
}
