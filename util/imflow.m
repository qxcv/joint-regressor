function out_flow = imflow(path_1, path_2)
%Compute optical flow between two images, specified by their paths
frame_1 = single(imread(path_1));
frame_2 = single(imread(path_2));
matches = deepmatching(frame_1, frame_2);
out_flow = deepflow2(frame_1, frame_2, matches);

