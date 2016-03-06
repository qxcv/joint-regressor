function save_centroid_pairwise_means(cache_dir, subpose_pa, shared_parts)
%SAVE_CENTROID_PAIRWISE_MEANS Compute subpose displacements and save.
dest_path = fullfile(cache_dir, 'subpose_disps.mat');
if ~exist(dest_path, 'file')
    centroid_path = fullfile(cache_dir, 'centroids.mat');
    centroids_m = load(centroid_path);
    centroids = centroids_m.centroids;
    subpose_disps = calc_subpose_disps(centroids, subpose_pa, shared_parts); %#ok
    save(dest_path, 'subpose_disps');
end
end