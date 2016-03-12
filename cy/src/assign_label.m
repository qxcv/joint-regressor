function labels = assign_label(all_data, clusters, pa, tsize)
assert(iscell(clusters) && length(clusters) == length(pa), ...
    'Need as many clusters as joints');
assert(ismatrix(clusters{1}), 'Centroids should be K*J matrix');
assert(all(cellfun(@(x) all(size(x) == size(clusters{1})), clusters)), ...
    'Size of clusters should be uniform');
K = size(clusters{1}, 1);
is_check = false;
% add mix field to imgs
p_no = numel(pa);
get_cell = @() cell(all_data.num_pairs, 1);
labels = struct('mix_id', get_cell(), ...
  'global_id', get_cell(), ...
  'near', get_cell(), ...
  'invalid', get_cell());
[nbh_IDs, global_IDs] = get_IDs(pa, K);

assert(false, 'You need to fix the rest of assign_label');

parfor pair_idx = 1:all_data.num_pairs
  labels(pair_idx).mix_id = cell(p_no, 1);
  labels(pair_idx).near = cell(p_no, 1);
  labels(pair_idx).invalid = false(p_no, 1);
  for p = 1:p_no
    nbh_N = numel(clusters{p});
    labels(pair_idx).mix_id{p} = zeros(nbh_N, 1, 'int32');
    labels(pair_idx).near{p} = cell(nbh_N, 1);
    invalid = false;
    for n = 1:nbh_N
      % find nearest
      nbh_idx = nbh_IDs{p}(n);
      if ( isfield(all_data, 'invalid') && (all_data(pair_idx).invalid(p) || all_data(pair_idx).invalid(nbh_idx)) )
        invalid = true;
      end
      cluster_num = numel(clusters{p}{n});
      centers = zeros(cluster_num, 2);
      for k = 1:cluster_num
        centers(k,:) = clusters{p}{n}(k).center;
        % sigmas(k,:) = clusters{p}{n}(k).sigma;
      end
      rp = norm_rp(all_data(pair_idx), p, nbh_idx, tsize);
      
      dists = bsxfun(@minus, centers, rp);
      dists = sqrt(sum(dists .^ 2, 2));
      
      [~,id] = min(dists,[],1);
      % for debug
      if (is_check && ~invalid)
        is_imgid = clusters{p}{n}(id).imgid;
        assert(is_imgid(pair_idx));
      end
      labels(pair_idx).mix_id{p}(n) = int32(id);
      labels(pair_idx).near{p}{n} = dists < 3 * dists(id);
    end
    % invalid
    labels(pair_idx).invalid(p) = invalid;
  end
  labels(pair_idx).global_id = int32(mix2global(labels(pair_idx).mix_id, global_IDs));
end

function global_id = mix2global(mix_id, global_IDs)
p_no = numel(mix_id);
global_id = zeros(p_no, 1);
for p = 1:p_no
  mixs = ones(3, 1);
  for ii = 1:numel(mix_id{p})
    mixs(ii) = mix_id{p}(ii);
  end
  global_id(p) = global_IDs{p}(mixs(1),mixs(2),mixs(3));
end
