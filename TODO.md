1. Check that I've subtracted mean pixel from current train/val patches.
2. Make sure that mean pixel subtraction is part of the automated pipeline.
3. Put centroid calculation into automated pipeline.
4. Increase loss_weight of poselet classification to 10-100 range (100 for
   parity with regressor loss; that may not be a good idea).
5. Cache centroids.
6. Get Keras up and running; need equivalent to current VGGNet.
7. Rewrite output code to write to space-saving Keras-compatible format (i.e.
   half-precision floats for flow & 24BPP for images).
