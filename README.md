# joint-regressor

This is the code for my 2016 COMP3710 project on human pose estimation. To run
it, fire up Matlab and use one of the following:

```matlab
>>> demo_mpii        % Train and test on MPII Cooking Activities
>>> demo_flic_piw    % Train on FLIC and and test on Poses in the Wild
>>> demo_h36m_upper  % Train and test on Human3.6M, using only upper body
```

Datasets are not included with this archive. However, they will be downloaded
automatically by the `demo_*` scripts.

Some important caveats to note:

- CNN training and evaluation requires a GPU with CUDA support and a significant
  amount of memory (12GB K80 and K40 GPUs were used for experiments).
- Training is also taxing on disk space and memory. Several hundred gigabytes of
  disk storage are required for the CNN training set, and some stages of the
  pipeline can take gigabytes of main memory.
- The training and testing process typically takes *several days*, most of which
  is spent training the CNN and structural SVM. You can save some time by
  interrupting CNN training part way through, but you will have to manually
  upgrade the partially trained model to a fully convolutional network and put
  the `cnn_model.json` (network specification) and `cnn_model.h5` (weights) files
  for the fully convolutional network in the cache. You will likely want to use
  the IPython notebook at `keras/upgrading-convnets.ipynb` to do this.
- Due to its high resource requirements, I have only been able to find one
  machine on which to test this code. Thus, it may break if moved to another
  machine.

## Originality

Since this code is being submitted for assessment, I need to document where it
came from. In a nutshell, all code in this directory was written by me, this
semester, with the following exceptions:

- Code in the `cy/` directory is adapted from Chen & Yuille's *Articulated Pose
  Estimation by a Graphical Model with Image Dependent Pairwise Relations*
  paper.
- Code in the `cmas/` directory is adapted from Cherian et al.'s *Mixing
  Body-Part Sequences for Human Pose Estimation*.
- Except for `get_deps` and the `flow` sub-directory, everything in `ext/` is a
  third-party dependency.
