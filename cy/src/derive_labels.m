function labels = derive_labels(cachedir, pa, imdata, clusters, subposes, K)
% derive pairwise relational type labels
label_path = fullfile(cachedir, 'labels.mat');

try
  load(label_path);
catch
  % assign mix
  labels = assign_label(imdata, pa, clusters, subposes, K);
  save(label_path, 'labels', '-v7.3');
end