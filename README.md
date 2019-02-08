# w2v 

* **Version:** 0.1-0
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Project home**: https://github.com/rbigdata/w2v
* **Bug reports**: https://github.com/rbigdata/w2v/issues


R bindings for an MPI+threads parallel word2vec. The internals are a modified version of [pWord2Vec](https://github.com/IntelLabs/pWord2Vec).


## Installation

In order to install the package, you will need a system installation of MPI. See the [pbdMPI package documentation]() for more information. You will also need the development versions of the float and pbdMPI packages:

```r
remotes::install_github("wrathematics/float")
remotes::install_github("snoweye/pbdMPI")
```

The development version is maintained on GitHub:

```r
remotes::install_github("RBigData/w2v")
```



## (un)Known Issues

To build the package at this time, you will need a fairly modern version of icc or gcc (I am using gcc-7). You should also have OpenBLAS or MKL installed and in your `$LD_LIBRARY_PATH` for the best performance.

It might not build on a mac. It definitely won't build on windows.


## Parameters

The default parameters are determined by the convenience function `w2v_params()`, which is the default argument for `params` in `w2v()`. You can modify a parameter by specifying it in `w2v_params()`. For example, if you want to set the initial learning rate to 0.01 and the number of training iterations to 10, you could do:

```r
params = w2v::w2v_params(alpha=0.01, iter=10)
```

And then pass the above to the `params` argument of `w2v()`. This all sounds more complicated than it is. See `?w2v::w2v_params` for an explanation of the available parameters.


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
Alpha: 0.000010  Progress: 100.00%  Words/sec: 476.16k  Words sync'ed: 65536
Wall Time: 179.10s

### Saving model
Wall Time: 2.59s
```

And inspecting the output file:

```
$ head -n 2 /tmp/out.txt 
71291 100
</s> 0.004003 0.004419 -0.003830 -0.003278 0.001367 0.003021 0.000941 0.000211 -0.003604 0.002218 -0.004356 0.001250 -0.000751 -0.000957 -0.003316 -0.001882 0.002579 0.003025 0.002969 0.001597 0.001545 -0.003803 -0.004096 0.004970 0.003801 0.003090 -0.000604 0.004016 -0.000495 0.000735 -0.000149 -0.002983 0.001312 -0.001337 -0.003825 0.004754 0.004379 -0.001095 -0.000226 0.000509 -0.003638 -0.004007 0.004555 0.000063 -0.002582 -0.003042 -0.003076 0.001697 0.000201 0.001331 -0.004214 -0.003808 -0.000130 0.001144 0.002550 -0.003170 0.004080 0.000927 0.001120 -0.000608 0.002986 -0.002288 -0.002097 0.002158 -0.000753 0.001031 0.001805 -0.004089 -0.001983 0.002914 0.004232 0.003932 -0.003047 -0.002108 -0.000909 0.002001 -0.003788 0.002998 0.002788 -0.001599 -0.001552 -0.002238 0.004229 0.003912 -0.001180 0.004215 0.004820 0.001815 0.004983 -0.003111 -0.001532 -0.002107 -0.002907 0.002815 0.001579 0.000425 -0.002194 0.001524 0.003059 0.000194 
```

We can easily read this into R:

```r
x = data.table::fread("/tmp/out.txt", sep=" ", skip=1, header=FALSE)
x[, 1:5]
##                  V1        V2        V3        V4        V5
##     1:         </s>  0.004003  0.004419 -0.003830 -0.003278
##     2:          the  0.102287 -0.274286 -0.010839  0.186329
##     3:           of  0.127809 -0.047539 -0.065880  0.262200
##     4:          and  0.043331 -0.094926  0.075654  0.108173
##     5:          one  0.088950 -0.107013 -0.169995  0.107114
##    ---                                                     
## 71287: snaggletooth  0.258485 -0.135700 -0.043408  0.213208
## 71288:   introvigne  0.098310  0.022797  0.060727  0.024171
## 71289:    denishawn  0.062993 -0.341420 -0.044524  0.225393
## 71290:      tamiris -0.168609 -0.229003  0.039223  0.087821
## 71291:    dolophine  0.185319 -0.038160 -0.048233  0.039484
```
