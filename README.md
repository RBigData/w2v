# w2v 

* **Version:** 0.1-0
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Project home**: https://github.com/rbigdata/w2v
* **Bug reports**: https://github.com/rbigdata/w2v/issues


R bindings for a MPI+threads parallel word2vec. The internals are a modified version of [pWord2Vec](https://github.com/IntelLabs/pWord2Vec).


## Installation

The development version is maintained on GitHub:

```r
remotes::install_github("RBigData/w2v")
```

You will need a system installation of MPI as well as the pbdMPI package.


## (un)Known Issues

To build the package at this time, you will need a fairly modern version of gcc (I am using gcc-7). You should also have OpenBLAS or MKL installed and in your `$LD_LIBRARY_PATH` for the best performance.

It might not build on a mac. It definitely won't build on windows.


## Example Use

We'll be using the [text8 dataset](http://mattmahoney.net/dc/textdata), which you can download here [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip). After downloading, uncompress it. Mine is stored in `/tmp/text8`.

The main driver is `w2v()`. You can either compute the vocabulary on the fly, or pre-compute it with `get_vocab()`. For a big problem, I would recommend pre-computing, since you may need to tweak the `w2v()` function arguments (not the word2vec params, the other ones) to get optimal performance. You can run this in an interactive R session:

```r
tf = "/tmp/text8"
vf = "/tmp/vocab.txt"
w2v::get_vocab(train_file=tf, vocab_file=vf, verbose=TRUE)
```

Running the above (with `verbose=TRUE`), I see:

```
### Generating vocabulary
Vocab size: 253855
Words in train file: 17005206
Wall Time: 7.14s

### Saving vocabulary
Wall Time: 0.07s
```

Next, save the following script as, say, `w2v.r` and run it with `mpirun -np 2 Rscript w2v.r`:

```r
suppressMessages(library(w2v))

tf = "~/tmp/pWord2Vec-master/data/text8"
vf = "/tmp/vocab.txt"
of = "/tmp/out.txt"

w2v(train_file=tf, output_file=of, vocab_file=vf, verbose=TRUE, nthreads=4)

finalize()
```

Here's what I see when I run it on my laptop:

```
### Reading vocabulary
Vocab size: 71291
Words in train file: 16718843
Wall Time: 0.31s

### Training
Alpha: 0.000010  Progress: 100.02%  Words/sec: 434.31k  Words sync'ed: 65536
Wall Time: 192.93s

### Saving model
Wall Time: 0.26s
```
