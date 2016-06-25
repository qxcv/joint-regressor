function map_locs = image_to_map_loc(locations, pyra_level, cnnpar)
%IMAGE_TO_MAP_LOC Convert (scaled) image locations to heatmap locations
% By "scaled", I mean that the image locations are scaled to the scale of
% the current pyramid level. Yes, that IS confusing.
% Locations should give pixel coordinates of top left corner of relevant
% subpose.
assert(ismatrix(locations) && size(locations, 2) == 2 && isstruct(pyra_level));
% Use pyra.{sizs,scale,pad}
pad_px = pyra_level.pad * cnnpar.step;
map_locs = floor((locations + pad_px - 1) / step) + 1;
assert(all(size(map_locs) == size(locations)));
end
