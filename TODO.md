1. Make dataset writing resumable. Probably do this by keeping track of
   which pairs I've written and which I haven't.
2. Write out scale factors for patches (e.g. 0.5 if the patch is half the
   size of the corresponding section of image it is taken from in the
   training set). This will help calculate accuracy correctly.
3. Consider scaling down images before doing augmentations. The scale could be
   chosen so that the largest crop we consider is still 224x224. At the moment,
   I suspect that everything is taking a huge amount of time because I'm trying
   to rotate/rescale/whatever colossal images.
4. Figure out why `end_evt` doesn't work in `train.py`. It seems like the
   workers are just ignoring it and thundering on anyway.
5. Consider changing the training process so that each run gets its own
   directory containing a complete model definition (in JSON) *and* a bunch of
   checkpointed weights. This would allow me to mess with `models.py` even
   while training, and not have to be confused when my net starts screwing up
   because I changed something subtle like the dropout probability.
6. Uh, the training code's batch count is totally broken. It seems to be adding
   the batch size to the running total after each epoch (so +48 each time, by
   default), when it really should be adding the number of batches in the epoch
   instead (+256, by default).
