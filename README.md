# joint-regressor

Another pose estimation project. Related to [my COMP2560
project](/qxcv/comp2560), but with a slightly different approach.

## Notes on MPII Cooking Activities data set

Data was downloaded from
[here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/human-activity-recognition/mpii-cooking-activities-dataset/).
Anoop mentioned that the continuous pose dataset was released because the
original data set was not sufficient to evaluate on. The [original
dataset](http://datasets.d2.mpi-inf.mpg.de/MPIICookingActivities/poseChallenge-1.1.zip)
is the one above the continuous pose challenge data set (labelled "pose
challenge" or something) which only included ~1000 training images (continuous)
and a few test/validation images (not continuous).

Since the continuous pose challenge isn't split into training, validation and
test sets, Anoop suggested that I use the entire thing for training and then
validate on the (continuous) test set of the original, non-continuous pose
estimation challenge.
