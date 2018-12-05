#' w2v_params
#' 
#' Generates the parameters list for use in \code{w2v()}.
#' 
#' @param binary
#' (boolean) Save the resulting vectors in binary mode?
#' @param disk
#' (boolean) Stream text from disk during training? Otherwise the text will be loaded memory before training.
#' @param negative
#' (non-negative integer) Number of negative examples. Common values are 3 - 10 (0 = not used).
#' @param iter
#' (positive integer) Number of training iterations.
#' @param window
#' (non-negative integer) Max skip length between words.
#' @param batch_size
#' (positive integer) The batch size used for mini-batch training.
#' @param min_count
#' (non-negative integer) Discard words that appear less than \code{min_count} times.
#' @param hidden_size
#' (positive integer) Size of word vectors.
#' @param min_sync_words
#' (positive integer) Minimal number of words to be synced at each model sync.
#' @param full_sync_times
#' (non-negative integer) Enforced full model sync-up times during training.
#' @param alpha
#' (float) Initial learning rate.
#' @param sample
#' (float) Threshold for word occurrence.
#' @param model_sync_period
#' (float) Synchronize model every \code{model_sync_period} seconds.
#' 
#' @return
#' A \code{w2v_params} object, for use in \code{w2v()}.
#' 
#' @export
w2v_params = function(
  binary = FALSE,
  disk = FALSE,
  negative = 5,
  iter = 5,
  window = 5,
  batch_size = 11,
  min_count = 5,
  hidden_size = 100,
  min_sync_words = 1024,
  full_sync_times = 0,
  alpha = 0.1,
  sample = 1e-3,
  model_sync_period = 0.1
)
{
  check.is.flag(binary)
  check.is.flag(disk)
  check.is.natnum(negative)
  check.is.posint(iter)
  check.is.natnum(window)
  check.is.posint(batch_size)
  check.is.natnum(min_count)
  check.is.posint(hidden_size)
  check.is.posint(min_sync_words)
  check.is.natnum(full_sync_times)
  check.is.float(alpha)
  check.is.float(sample)
  check.is.float(model_sync_period)
  
  p = list(
    binary = as.logical(binary),
    disk = as.logical(disk),
    negative = as.integer(negative),
    iter = as.integer(iter),
    window = as.integer(window),
    batch_size = as.integer(batch_size),
    min_count = as.integer(min_count),
    hidden_size = as.integer(hidden_size),
    min_sync_words = as.integer(min_sync_words),
    full_sync_times = as.integer(full_sync_times),
    alpha = as.double(alpha),
    sample = as.double(sample),
    model_sync_period = as.double(model_sync_period)
  )
  
  class(p) = "w2v_params"
  p
}
