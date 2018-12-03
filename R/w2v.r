#' @useDynLib w2v R_w2v
#' @export
w2v = function(train, verbose=FALSE)
{
  train = tools::file_path_as_absolute(train)
  .Call(R_w2v, train, verbose)
}
