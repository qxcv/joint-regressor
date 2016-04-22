function model = build_model(subpose_pa, K, subpose_disps, cnn_conf, mean_pixels, ...
    pyra_scales, tsize, memsize, template_scale)
% This function merges together separate part models into a tree structure

[~, global_IDs, ~] = get_IDs(subpose_pa, K);

% parameters need for cnn
model.cnn = cnn_conf;
model.cnn.mean_pixels = mean_pixels;
model.cnn.cnn_output_dim = global_IDs{end}(end);
model.cnn.psize = tsize * cnn_conf.step;

% Factor by which we expan bounding boxes for parts before taking crop for
% CNN (well virtual crop here, since we're using a fully convolutional
% CNN).
model.template_scale = template_scale;

model.memsize = memsize;

model.tsize = tsize;
model.K = K;
model.subpose_disps = subpose_disps;

% bias
model.bias    = struct('w',{},'i',{});
% appearance paramters
model.apps = struct('w',{},'i',{});
% deformation gaussian parameters
model.gaus    = struct('w',{},'i',{});

model.components = struct('parent',{}, 'pid', {}, 'subpose_disps', {}, ...
    'biasid', {}, 'appid', {}, 'app_global_ids', {}, 'gauid', {});

model.subpose_pa = subpose_pa;
model.pyra_scales = pyra_scales;
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
    
    if parent == 0
        assert(isempty(model.root));
        model.root = subpose_idx;
    end
    
    % Will be K*K*2 matrix; use like disp = p.subpose_disps(child_type,
    % parent_type, :) to get child_shared_endpoint_loc -
    % parent_shared_endpoint_loc
    if parent ~= 0
        disps = squeeze(subpose_disps(subpose_idx, :, :, :));
        child_K = size(disps, 1);
        parent_K = size(disps, 2);
        p.subpose_disps = cell([1 child_K]);
        % Slow, but whatever
        for child_type=1:child_K
            p.subpose_disps{child_type} = mat2cell(....
                squeeze(disps(child_type, :, :)), ones([1 parent_K]), 2);
        end
    else
        p.subpose_disps = {};
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
    % TODO: Should update these initial scores to be of the correct
    % order-of-magnitude for my model.
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
assert(model.root == 1, 'Root should be first element');
end
