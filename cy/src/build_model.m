function model = build_model(subpose_pa, K, subpose_disps, cnn_conf, mean_pixels, ...
    interval, tsize, memsize)
% This function merges together separate part models into a tree structure

[nbh_IDs, global_IDs, target_IDs] = get_IDs(subpose_pa, K);

% parameters need for cnn
model.cnn = cnn_conf;
model.cnn.mean_pixels = mean_pixels;
model.cnn.cnn_output_dim = global_IDs{end}(end);
model.cnn.psize = tsize * cnn_conf.step;

model.memsize = memsize;

model.tsize = tsize;
model.global_IDs = global_IDs;
model.nbh_IDs = nbh_IDs;
model.target_IDs = target_IDs;
model.K = K;
model.subpose_disps = subpose_disps;

% bias
model.bias    = struct('w',{},'i',{});
% appearance paramters
model.apps = struct('w',{},'i',{});
% deformation gaussian parameters
model.gaus    = struct('w',{},'i',{});

model.components = struct('parent',{}, 'pid', {}, 'nbh_IDs', {}, ...
  'subpose_disps', {}, 'biasid', {}, 'appid', {}, 'app_global_ids', {}, ...
  'gauid', {});

model.subpose_pa = subpose_pa;
model.interval = interval;
model.sbin = cnn_conf.step;
model.len = 0;
model.root = [];

% add children
for subpose_idx = 1:length(subpose_pa)
    child = subpose_idx;
    parent = subpose_pa(child);
    assert(parent < child, 'Parents array should be toposorted');
    % p is a single struct array entry which will get appended to
    % components{1}.
    p.parent = parent;
    p.pid    = child;
    p.nbh_IDs = nbh_IDs{p.pid};
    
    if parent == 0
        assert(isempty(model.root));
        model.root = subpose_idx;
    end
    
    % Will be K*K*2 matrix; use like disp = p.subpose_disps(child_type,
    % parent_type, :) to get child_center - parent_center (IIRC)
    if parent ~= 0
        p.subpose_disps = squeeze(subpose_disps(subpose_idx, :, :, :));
        assert(ndims(p.subpose_disps) == 3);
        assert(size(p.subpose_disps, 3) == 2);
    else
        p.subpose_disps = [];
    end
    
    % add bias (only to root, i.e. parent == 0)
    p.biasid = [];
    if parent == 0
        nb  = length(model.bias);
        b.w = 0;
        b.i = model.len + 1;
        model.bias(nb+1) = b;
        model.len = model.len + numel(b.w);
        p.biasid = nb+1;
    end
    
    % add appearance parameters
    p.appid = [];
    
    nf  = length(model.apps);
    % Yep, there's only one appearance term, even though *my* appearance term
    % also considers part type.
    f.w = 0.01;                     % encourage larger appearance score
    f.i = model.len + 1;
    
    model.apps(nf+1) = f;
    model.len = model.len + numel(f.w);
    
    p.appid = [p.appid, nf+1];
    p.app_global_ids = global_IDs{subpose_idx};
    % add gaussian parameters (for spring deformation)
    p.gauid = [];
    if parent ~= 0
        % Skip root node because we want one set of weights per edge
        ng  = length(model.gaus);
        g.w = [0.01, 0, 0.01, 0]; % [dx^2, dx, dy^2, dy]the normalization factor + variance for (x,y)
        % res = [-dx^2, -dx, -dy^2, -dy]';
        g.i = model.len + 1;
        
        model.gaus(ng+1) = g;
        model.len = model.len + numel(g.w);
        p.gauid = ng+1;
    end
    
    np = length(model.components);
    model.components(np+1) = p;
end

assert(isscalar(model.root), 'Could not find root part');
% assert(model.root == 1, 'Root should be first element'); XXX
end