1. Increase `loss_weight` of poselet classification to 10-100 range (100 for
   parity with regressor loss; that may not be a good idea).
2. Cache centroids.
3. Complete the flow experiment Anoop suggested.
4. Consider scaling down images before doing augmentations. The scale could be
   chosen so that the largest crop we consider is still 224x224. At the moment,
   I suspect that everything is taking a huge amount of time because I'm trying
   to rotate/rescale/whatever colossal images.
