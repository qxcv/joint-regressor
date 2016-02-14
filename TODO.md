1. Increase loss_weight of poselet classification to 10-100 range (100 for
   parity with regressor loss; that may not be a good idea).
2. Cache centroids.
3. Rewrite code so that it looks at the SECOND number in the continuous pose
   dataset. At the moment it is looking at the first number, which is actually
   the index of the image /in the activity track/ (which includes a superset of
   the images in the continuous pose estimation track). I have no idea why that
   is listed first, but it is.
