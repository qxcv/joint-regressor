function pose_dets = stitch_all_seqs(pair_dets, stitch_weights, valid_parts, cache_dir)
%STITCH_ALL_SEQS Stitch several sequences in parallel
dest = fullfile(cache_dir, 'stitched_seqs.mat');
try
    l = load(dest);
    pose_dets = l.pose_dets;
    fprintf('Loaded stitched detections from %s\n', dest);
catch
    fprintf('Re-stitching detections\n');
    pose_dets = cell([1 length(pair_dets)]);
    parfor i=1:length(pair_dets)
        fprintf('Re-stitching sequence %i/%i\n', i, length(pair_dets));
        pairs = pair_dets{i};
        pose_dets{i} = stitch_seq(pairs, stitch_weights, valid_parts);
    end
    save(dest, 'pose_dets');
end
end

