# joint-regressor

Another pose estimation project. Related to [my COMP2560
project](/qxcv/comp2560), but with a slightly different approach.

## A note on dependencies

This project depends on [DeepMatching](http://lear.inrialpes.fr/src/deepmatching/)
and [DeepFlow](http://lear.inrialpes.fr/src/deepflow/). I have included a
script (in `ext/get_deps.m`) which attempts to download and compile these
dependencies. However, they can still fail if the wrong BLAS version is
used (e.g. OpenBLAS is used where OpenBLAS is not present). To compile
manually after a failure, `cd` into `ext/DeepFlow_release<ver>` and build
manually (including mex wrapper), then touch
`ext/DeepFlow_release<ver>/.built`. You can do the same for DeepMatching.

If Matlab segfaults while computing flow, then you'll need to `LD_PRELOAD`
the appropriate BLAS library (e.g.
`LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_rt.so` for MKL).