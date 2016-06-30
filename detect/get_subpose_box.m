function box = get_subpose_box(heatmap_x, heatmap_y, det_side, pad, scale)
%GET_SUBPOSE_BOX Grab the bounding box associated with a subpose detection
% at the given location in the output heatmap. det_side is cnn_edge_length
% / cnn_step, pad is the padding added to the original image before
% producing the heatmap, divided by the step, scale is the factor by which
% the heatmap needs to be multiplied to get back to the original image
% resolution.

% RV is in original image coordinates, so if the detection is perfect then
% orig_image(y1:y2, x1:x2, :) will select exactly the region containing the
% subpose.

x1 = (heatmap_x - 1 - pad)*scale+1;
y1 = (heatmap_y - 1 - pad)*scale+1;
x2 = x1 + det_side*scale - 1;
y2 = y1 + det_side*scale - 1;
box = [x1 y1 x2 y2];
end
