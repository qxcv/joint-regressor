function labels = derive_labels(cachedir, pa, imdata, subposes, edge_means, tsize)
% derive pairwise relational type labels
label_path = fullfile(cachedir, 'labels.mat');

try
  load(label_path);
catch
  % assign mix
  labels = assign_label(imdata, pa, subposes, edge_means, tsize);
  save(label_path, 'labels', '-v7.3');
end