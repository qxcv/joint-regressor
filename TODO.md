1. Make dataset writing resumable. Probably do this by keeping track of
   which pairs I've written and which I haven't.
2. Write out scale factors for patches (e.g. 0.5 if the patch is half the
   size of the corresponding section of image it is taken from in the
   training set). This will help calculate accuracy correctly.
3. Consider scaling down images before doing augmentations. The scale couldbe
   chosen so that the largest crop we consider is still 224x224. At the moment,
   I suspect that everything is taking a huge amount of time because I'm trying
   to rotate/rescale/whatever colossal images.