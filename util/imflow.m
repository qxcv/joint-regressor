function out_flow = imflow(path_1, path_2)
%IMFLOW Compute optical flow between two images, specified by their paths
% Note that we *only* open frame_1 to get its width and height
frame_1 = single(imread(path_1));
[w, h, ~] = size(frame_1);
para = get_para_flow(w, h);
[out_flow, ~, ~] = LDOF(path_1, path_2, para, 0);