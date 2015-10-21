function sub_mp(h5_path, mean_pixel)
%SUB_MP Subtracts a mean pixel from an HDF5 file. This is only useful if
% you fucked up and forgot to subtract it to start with.
data = h5read(h5_path, '/data');
chans_first = permute(data, [3, 2, 1, 4]);
subbed = bsxfun(@minus, chans_first, mean_pixel);
prop_order = permute(subbed, [3, 2, 1, 4]);
h5write(h5_path, '/data', prop_order);
end

