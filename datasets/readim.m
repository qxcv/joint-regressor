% Like imread, but for data set samples. This function will be useful later if I
% want to do transformations (e.g. I can set a .flip flag on samples which I
% want to flip, like Chen & Yuille do).
function im = readim(datum)
im = imread(datum.image_path);
