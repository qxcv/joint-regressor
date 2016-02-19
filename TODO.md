1. Make dataset writing resumable. Probably do this by keeping track of
   which pairs I've written and which I haven't.
2. Write out scale factors for patches (e.g. 0.5 if the patch is half the
   size of the corresponding section of image it is taken from in the
   training set). This will help calculate accuracy correctly.
3. Consider scaling down images before doing augmentations. The scale could be
   chosen so that the largest crop we consider is still 224x224. At the moment,
   I suspect that everything is taking a huge amount of time because I'm trying
   to rotate/rescale/whatever colossal images.
4. `train.py` is becoming unwieldy as I introduce more features (first
   configurable RGB/flow data, then on-the-fly mean pixel subtraction, and so
   on). I think I could significantly reduce this complexity if I followed a
   Caffe-style approach of reading inputs and mean pixels straight from
   correspondingly named datasets in an HDF5 file, which means that the
   complexity of whether to merge flow and RGB data, how to do mean
   subtraction, etc. can be moved into configuration rather than having to
   update the actual DS fetching code each time something needs to be updated
   :/
5. Figure out why `end_evt` doesn't work in `train.py`. It seems like the
   workers are just ignoring it and thundering on anyway.
