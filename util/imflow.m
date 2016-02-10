function out_flow = imflow(path_1, path_2)
%IMFLOW Compute optical flow between two images, specified by their paths
% Note that we *only* open frame_1 to get its width and height
frame_1 = imread(path_1);
frame_2 = imread(path_2);
out_flow = broxOpticalFlow(frame_1, frame_2);
end