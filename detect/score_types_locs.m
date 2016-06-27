function rv_score = score_types_locs(types, locs, unaries, model)
%SCORE_TYPES_LOCS Score types and locations for subposes
assert(isvector(types) && ismatrix(locs) && size(locs, 2) == 2 ...
    && iscell(unaries) && isstruct(model));

sbin = model.sbin;
[components, app_weight_cell] = modelcomponents(model);
num_sp = length(components);

% Extract features (appearance terms and deformation gaussians)
apps = nan([1, num_sp]);
gaus = nan([num_sp - 1, 4]);
app_weights = apps;
gau_weights = gaus;
for sp_idx=1:num_sp
    loc = locs(sp_idx, :);
    x = loc(1); y = loc(2); t = types(sp_idx);
    unary_map = unaries{sp_idx}(:, :, t);
    unary_width = size(unary_map, 2);
    unary_height = size(unary_map, 1);
    if any([x y] < 1 | [x y] > [unary_width unary_height])
        % If the point is out of bounds, we give it a really low score just
        % to discourage it (may have to make this faster);
        apps(sp_idx) = min(log(1e-10), min(unary_map(:)));
    else
        apps(sp_idx) = unary_map(y, x);
    end
    app_weights(sp_idx) = app_weight_cell{sp_idx};
    
    if sp_idx > 1
        part = components(sp_idx);
        disps = part.subpose_disps;
        parent_idx = part.parent;
        assert(0 < parent_idx && parent_idx < sp_idx);
        p_loc = locs(parent_idx, :);
        p_x = p_loc(1); p_y = p_loc(2); p_t = types(parent_idx);
        gaus(sp_idx-1, :) = defvector(disps, x, y, p_x, p_y, t, p_t, sbin);
        gau_weights(sp_idx-1, :) = part.gauw;
    end
end

% Doing this all in one step for transparency :)
bias = model.bias.w;
rv_score = bias + dot(app_weights, apps) + sum(dot(gau_weights, gaus));
assert(~any(isnan(rv_score)));
end

