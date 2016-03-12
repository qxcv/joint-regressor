function labels = derive_labels(cachedir, pa, clusters, imdata, tsize)
% derive pairwise relational type labels
label_path = fullfile(cachedir, 'labels.mat');

try
  load(label_path);
catch
  % assign mix
  labels = assign_label(imdata, clusters, pa, tsize);
  save(label_path, 'labels', '-v7.3');
end