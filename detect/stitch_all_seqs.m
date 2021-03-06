function pose_dets = stitch_all_seqs(pair_dets, num_stitch_dets, ...
    stitch_weights, valid_parts, cache_dir)
%STITCH_ALL_SEQS Stitch several sequences in parallel
should_cache = exist('cache_dir', 'var');

if should_cache
    dest = fullfile(cache_dir, 'stitched_seqs.mat');
    try
        l = load(dest);
        pose_dets = l.pose_dets;
        fprintf('Loaded stitched detections from %s\n', dest);
        return
    catch
        fprintf('Re-stitching detections\n');
    end
end

pose_dets = cell([1 length(pair_dets)]);
for i=1:length(pair_dets) % XXX: Reinstate parfor
    fprintf('Re-stitching sequence %i/%i\n', i, length(pair_dets));
    pairs = pair_dets{i};
    pose_dets{i} = stitch_seq(pairs, num_stitch_dets, stitch_weights, ...
        valid_parts);
end

if should_cache
    save(dest, 'pose_dets');
end
end

