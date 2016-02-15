1. Increase `loss_weight` of poselet classification to 10-100 range (100 for
   parity with regressor loss; that may not be a good idea).
2. Cache centroids.
3. Use MPII cooking actions pose challenge training data set (EIGHT NOUN
   PILEUP!) as source for validation data and ensure that said data is properly
   scene-segmented.
4. Change train.py so that field names are configurable. Doing this and then
   writing a new field for the relevant joint subset seems to be the most
   elegant way of supporting one-joint-only regression (e.g. wrist-only
   regression for the flow experiment).
5. Complete the flow experiment Anoop suggested.
