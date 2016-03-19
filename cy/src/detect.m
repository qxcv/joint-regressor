function [boxes,model,ex] = detect(im1_info, im2_info, model, thresh, ...
    bbox, overlap, id, label)
% Detect objects in image using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part
%
% If bbox is not empty, we pick best detection with significant overlap.
% If label is included, we write feature vectors to a global QP structure
%
% This function updates the model (by running the QP solver) if upper and
% lower bound differs
%
% detect() is called from two places in train.m. The signatures of those
% calls are:
%                         1       2      3   4   5  6  7
% 1) [box,model] = detect(neg(i), model, -1, [], 0, i, -1);
%                 1        2      3  4     5        6   7
% 2) box = detect(pos(ii), model, 0, bbox, overlap, ii, 1);

% XXX: List of things I've found which are broken and will need changing:
% 1) Need to change message-passing code to account for types of each node.
% This probably means that some previously 2D arrays used in message
% passing become 3D, or even higher-dimensional (because now we need to
% remember which type of each part yielded the highest score). Remember
% that I need to recover joint locations at the end of this process, and
% that will require access to type information!
% 2) In passmsg, I'll have to figure out how to compute displacements
% correctly. In particular, I'll have to make sure that displacement sign
% is correct and that the displacement is scaled correctly so that it is in
% "heatmap coordinates".
% 3) There are a few places where I need to find bounding boxes for parts
% (well, sub-poses in my model). I should do that by using the CNN
% receptive field corresponding to each sub-pose, instead of whatever
% heuristic measure is currently used.

INF = 1e10;

if nargin > 3 && ~isempty(bbox)
    latent = true;
    if label > 0
        thresh = -INF;
    end
else
    latent = false;
end

% Compute the feature pyramid and prepare filter
im1 = readim(im1_info);
im2 = readim(im2_info);
flow = imflow(im1_info.image_path, im2_info.image_path);
im_stack = cat(3, im1, im2);
% if has box information, crop it
assert(false, 'Need to fix the rest of detect');
if latent && label > 0
    % crop positive images to speed up latent search
    [im_stack, flow, bbox] = cropscale_pos(im_stack, flow, bbox, model.cnn.psize);
end

[pyra, unary_map] = imCNNdet(im_stack, flow, model);

levels = 1:length(pyra);

% Define global QP if we are writing features
% Randomize order to increase effectiveness of model updating
write = false;
if nargin > 5
    global qp;
    write  = true;
    levels = levels(randperm(length(levels)));
end
if nargin < 6
    id = 0;
end
if nargin < 7
    label = 0;
end

% Cache various statistics derived from model
[components, apps] = modelcomponents(model);

boxes = zeros(100000, 4 * length(components) + 2);
cnt = 0;

ex.blocks = [];
ex.id = [label id 0 0 0];

if latent && label > 0
    % record best when doing latent on positive example
    best_ex = ex;
    best_box = [];
end

% Iterate over random permutation of scales and components,
for level = levels
    % Iterate through mixture components
    sizs = pyra(level).sizs;
    num_subposes = length(components);
    assert(num_subposes > 1, 'Only %d parts?\n', num_subposes);
    
    % Skip if there is no overlap of root filter with bbox
    if latent
        skipflag = 0;
        for subpose_idx = 1:num_subposes
            % because all mixtures for one part is the same size, we only need to do this once
            ovmask = testoverlap(components(subpose_idx).sizx(1), ...
                components(subpose_idx).sizy(1), sizs(1), sizs(2), ...
                pyra(level), bbox.xy(subpose_idx,:), overlap);
            if ~any(ovmask)
                skipflag = 1;
                break;
            end
        end
        if skipflag == 1
            continue;
        end
    end
    % Local scores
    
    for subpose_idx = 1:num_subposes
        components(subpose_idx).appMap = unary_map{level}{subpose_idx};
        assert(ndims(components(subpose_idx).appMap) == 3, ...
            'Need h*w*K unary map');
        
        % appid will be 1x1, and gives the unary weight associated with
        % this subpose
        f = components(subpose_idx).appid;
        assert(isscalar(f), 'Should have only one weight ID');
        assert(isscalar(apps{f}, 'Should have only one weight for that ID'));
        % .score will now be h*w*K for each part
        weighted_apps = components(subpose_idx).appMap * apps{f};
        assert(ndims(weighted_apps) == 3);
        assert(size(weihted_apps, 3) == model.K);
        components(subpose_idx).score = weighted_apps;
        components(subpose_idx).level = level;
        
        if latent
            assert(false, 'Need to fix this check to deal with bigger unaries');
            ovmask = testoverlap(components(subpose_idx).sizx, components(subpose_idx).sizy, ...
                sizs(1), sizs(2), pyra(level), bbox.xy(subpose_idx,:), overlap);
            assert(ismatrix(ovmask));
            tmpscore = components(subpose_idx).score;
            ovmask = repmat(ovmask, 1, 1, size(tmpscore, 3));
            assert(all(size(ovmask) == size(tmpscore)));
            % label supervision
            if label > 0
                tmpscore(~ovmask) = -INF;
                % mixture part supervision
                % Disabled for now because I don't have a .defMap
                %           if isfield(iminfo,'near')
                %             for n = 1:numel(parts(p).nbh_IDs)
                %               parts(p).defMap{n}(:,:,~iminfo.near{p}{n}) = -INF;
                %             end
                %           end
            elseif label < 0
                tmpscore(ovmask) = -INF;
            end
            assert(all(size(components(subpose_idx).score) == size(tmpscore)));
            components(subpose_idx).score = tmpscore;
        end
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for subpose_idx = num_subposes:-1:2
        child = components(subpose_idx);
        par_idx = components(subpose_idx).parent;
        parent = components(par_idx);
        
        [msg, components(subpose_idx).Ix, components(subpose_idx).Iy, ...
            components(subpose_idx).Im{limb_to_parent}, ...
            components(par_idx).Im{limb_to_child}] ...
            = passmsg(child, parent);
        components(par_idx).score = components(par_idx).score + msg;
    end
    
    % Add bias to root score
    % XXX: Will also need to account for root node type and its
    % corresponding appearance map.
    components(model.root).score = ...
        components(model.root).score + components(model.root).b;
    rscore = components(model.root).score;
    
    % keep the positive example with the highest score in latent mode
    if latent && label > 0
        thresh = max(thresh,max(rscore(:)));
    end
    
    [Y, X] = find(rscore >= thresh);
    % Walk back down tree following pointers
    % (DEBUG) Assert extracted feature re-produces score
    for i = 1:length(X)
        cnt = cnt + 1;
        x = X(i);
        y = Y(i);
        
        [box, ex] = backtrack(x, y, components, pyra(level), ex, write);
        
        boxes(cnt,:) = [box c rscore(y,x)];
        if write && (~latent || label < 0)
            qp_write(ex);
            qp.ub = qp.ub + qp.Cneg*max(1+rscore(y,x),0);
        elseif latent && label > 0
            if isempty(best_box)
                best_box = boxes(cnt,:);
                best_ex = ex;
            elseif best_box(end) < rscore(y,x)
                % update best
                best_box = boxes(cnt,:);
                best_ex = ex;
            end
        end
    end
    
    % Crucial DEBUG assertion:
    % If we're computing features, assert extracted feature re-produces
    % score (see qp_writ.m for computing original score)
    if write && (~latent || label < 0) && ~isempty(X) && qp.n < length(qp.a)
        w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
        assert(abs(score(w,qp.x,qp.n) - rscore(y,x)) < 1e-5);
    end
    
    % Optimize qp with coordinate descent, and update model
    if write && (~latent || label < 0) && ...
            (qp.lb < 0 || 1 - qp.lb/qp.ub > .05 || qp.n == length(qp.sv))
        model = optimize(model);
        [components, apps] = modelcomponents(model);
    end
end

boxes = boxes(1:cnt,:);

if latent && ~isempty(boxes) && label > 0
    boxes = best_box;
    if write
        qp_write(best_ex);
    end
end
end

% Backtrack through dynamic programming messages to estimate part locations
% and the associated feature vector
function [box,ex] = backtrack(x,y,parts,pyra,ex,write)
numparts = length(parts);
ptr = zeros(numparts,2);
box = zeros(numparts,4);
k   = 1;
p   = parts(k);
ptr(k, :) = [x, y];
scale = pyra.scale;
x1  = (x - 1 - pyra.padx)*scale+1;
y1  = (y - 1 - pyra.pady)*scale+1;
x2  = x1 + p.sizx*scale - 1;
y2  = y1 + p.sizy*scale - 1;

box(k,:) = [x1 y1 x2 y2];

if write
    ex.id(3:5) = [p.level round(x+p.sizx/2) round(y+p.sizy/2)];
    ex.blocks = [];
    ex.blocks(end+1).i = p.biasI;
    ex.blocks(end).x   = 1;
    f = parts(k).appMap(y, x);
    ex.blocks(end+1).i = p.appI;
    ex.blocks(end).x   = f;
end
for k = 2:numparts
    p   = parts(k);
    par = p.parent;
    
    x   = ptr(par,1);
    y   = ptr(par,2);
    
    ptr(k,1) = p.Ix(y,x);
    ptr(k,2) = p.Iy(y,x);
    
    x1  = (ptr(k,1) - 1 - pyra.padx)*scale+1;
    y1  = (ptr(k,2) - 1 - pyra.pady)*scale+1;
    x2  = x1 + p.sizx*scale - 1;
    y2  = y1 + p.sizy*scale - 1;
    box(k,:) = [x1 y1 x2 y2];
    
    if write
        cbid = find(p.nbh_IDs == parts(par).pid);
        pbid = find(parts(par).nbh_IDs == p.pid);
        
        cm = p.Im{cbid}(y,x);
        pm = parts(par).Im{pbid}(y,x);
        
        % two prior of deformation
        ex.blocks(end+1).i = p.pdefI(cbid);
        ex.blocks(end).x = p.defMap{cbid}(ptr(k,2),ptr(k,1),cm);
        
        % two deformations
        % XXX: This may be incorrect.
        ex.blocks(end+1).i = p.gauI{cbid}(cm);
        ex.blocks(end).x   = defvector(p, ptr(k,1),ptr(k,2),x,y,cm,cbid);
        
        ex.blocks(end+1).i = parts(par).gauI{pbid}(pm);
        ex.blocks(end).x   = defvector(parts(par),x,y,ptr(k,1),ptr(k,2),pm,pbid);
        
        x = ptr(k,1);
        y = ptr(k,2);
        
        % unary
        f = parts(k).appMap(y, x);
        ex.blocks(end+1).i = p.appI;
        ex.blocks(end).x = f;
    end
end
box = reshape(box', 1, 4*numparts);
end

% Update QP with coordinate descent
% and return the asociated model
function model = optimize(model)
global qp;
fprintf('.');
if qp.lb < 0 || qp.n == length(qp.a)
    qp_opt();
    qp_prune();
else
    qp_one();
end
model = vec2model(qp_w(), model);
end