function model = train(name, model, pos, neg, num_iters, C, wpos, maxsize, overlap)
% Train a structured SVM with latent assignement of positive variables
%                      1     2      3    4    5     6  7     8        9
% pos  = list of positive images with part annotations
% neg  = list of negative images
% iter is the number of training iterations
%   C  = scale factor for slack loss
% wpos =  amount to weight errors on positives
% maxsize = maximum size of the training data cache (in GB)
% overlap =  minimum overlap in latent positive search
%
%                                           1    2      3        4        5
% In train_model.m, this is called as train(cls, model, pos_val, neg_val, 1);
% Originally I thought this was called recursively (I think it might be in
% Y&R's code?), but a quick grep indicates that the only call is in
% train_model. This raises the question of why the functino takes nine
% arguments but only ever gets five of them.

model.name = name;

if ~exist('num_iters', 'var')
  num_iters = 1;
end

if ~exist('C', 'var')
  C = 0.002;
end

if ~exist('wpos', 'var')
  wpos = 2;
end

if ~exist('maxsize', 'var')
  % Estimated #sv = (wpos + 1) * # of positive examples
  % maxsize*1e9/(4*model.len)  = # of examples we can store, encoded as 4-byte floats
  no_sv = (wpos+1) * pos.num_pairs;
  maxsize = 10 * no_sv * 4 * sparselen(model) / 1e9;
  maxsize = min(max(maxsize, model.memsize),7.5);
end

fprintf('Using %.3f GB\n', maxsize);

if ~exist('overlap', 'var')
  overlap = 0.6; % at least 0.6 overlap to be positive
end

% Vectorize the model
len  = sparselen(model);
nmax = round(maxsize*.25e9/len);

% reset random seed to reproduce result
rng(0);

% Define global QP problem
clear global qp;
global qp;
% qp.x(:,i) = examples
% qp.i(:,i) = id
% qp.b(:,i) = bias of linear constraint
% qp.d(i)   = ||qp.x(:,i)||^2
% qp.a(i)   = ith dual variable
qp.x   = zeros(len, nmax, 'single');
qp.i   = zeros(6, nmax, 'int32');
qp.b   = ones(nmax, 1, 'single');
qp.d   = zeros(nmax, 1, 'double');
qp.a   = zeros(nmax, 1, 'double');
qp.sv  = false(1, nmax);
qp.n   = 0;
qp.lb = [];

[qp.w, qp.wreg, qp.w0, qp.noneg] = model2vec(model);
qp.Cpos = C*wpos;
qp.Cneg = C;
qp.w    = (qp.w - qp.w0).*qp.wreg;

for iter_idx=1:num_iters,
    fprintf('\niter: %d/%d\n', iter_idx, num_iters);
    
    qp.n = 0;
    numpositives = poslatent(name, iter_idx, model, pos, overlap);
    
    fprintf('got %d positives\n', numpositives);
    assert(qp.n <= nmax);
    
    % Fix positive examples as permanent support vectors
    % Initialize QP soln to a valid weight vector
    % Update QP with coordinate descent
    qp.svfix = 1:qp.n;
    qp.sv(qp.svfix) = 1;
    
    qp_prune();
    qp_opt();
    model = vec2model(qp_w(), model);
    
    % grab negative examples from negative images
    mining_onneg(model, neg, nmax)
    
    % One final pass of optimization
    qp_opt();
    model = vec2model(qp_w(),model);
    
    fprintf('\nDONE iter: %d/%d #sv=%d/%d, LB=%.4f\n', iter_idx, ...
        num_iters, sum(qp.sv), nmax, qp.lb);
    
    % Compute minimum score on positive example (with raw, unscaled features)
    r = sort(qp_scorepos());
    model.thresh = r(ceil(length(r)*.05));
    model.lb = qp.lb;
    model.ub = qp.ub;
end
fprintf('qp.x size = [%d %d]\n', size(qp.x));
clear global qp;
end

% negative mining on negative images
function numnegatives = mining_onneg(model, neg, nmax)
model.interval = 3;
numnegatives = 0;
global qp;
% grab negative examples from negative images
for neg_num = 1:neg.num_pairs
    fprintf('\n Image(%d/%d)', neg_num, neg.num_pairs);
    % last argument is a label for the pose. detect() will use this to update the
    % global qp (ugh).
    d1 = neg.data(neg.pairs(neg_num).fst);
    d2 = neg.data(neg.pairs(neg_num).snd);
    [box, model] = detect(d1, d2, model, -1, [], 0, neg_num, -1);
    numnegatives = numnegatives + size(box,1);
    fprintf(' #cache+%d=%d/%d, #sv=%d, #sv>0=%d, (est)UB=%.4f, LB=%.4f', ...
        size(box,1), qp.n, nmax, sum(qp.sv), sum(qp.a>0), qp.ub, qp.lb);
    
    % Stop if cache is full
    if sum(qp.sv) == nmax
        break;
    end
end
end

% get positive examples using latent detections
% we create virtual examples by flipping each image left to right
% XXX: where is the flipping part?
function numpositives = poslatent(name, t, model, pos, overlap)
num_pairs = pos.num_pairs;
% numpositives was length(model.components), which I think would have been
% 1 before I changed model.components from a 1x1 cell array containing a
% single struct array to just a struct array itself.
numpositives = 0;
minsize = prod(double(model.tsize*model.sbin));

for pair_num = 1:num_pairs
    fprintf('%s: iter %d: latent positive: %d/%d\n', name, t, pair_num, num_pairs);
    % skip small examples
    pair = pos.pairs(pair_num);
    scale = pair.scale;
    bbox_side = scale / 2;
    d1 = pos.data(pair.fst);
    d2 = pos.data(pair.snd);
    all_joints = cat(1, d1.joint_locs, d2.joint_locs);
    assert(ismatrix(all_joints) && size(all_joints, 2) == 2);
    bbox.xy = [all_joints(:,1)-bbox_side, all_joints(:,2)-bbox_side, ...
        all_joints(:,1)+bbox_side, all_joints(:,2)+bbox_side];
    area = (bbox.xy(:,3)-bbox.xy(:,1)+1).*(bbox.xy(:,4)-bbox.xy(:,2)+1);
    if any(area < minsize/1.5)
        % skip only when exmaple are too small
        continue;
    end
    
    % get example
    % note that detect is updating qp using ii and the label which we supply
    % it at the end, as above (but the label is 1 this time since we have a
    % positive)
    box = detect(d1, d2, pair, model, 0, bbox, overlap, pair_num, 1);
    if ~isempty(box)
        fprintf(' (sc=%.3f)\n', box(1, end));
        numpositives = numpositives+1;
    end
end
end

% Compute score (weights*x) on positives examples (see qp_write.m)
% Standardized QP stores w*x' where w = (weights-w0)*r, x' = c_i*(x/r)
% (w/r + w0)*(x'*r/c_i) = (v + w0*r)*x'/ C
function scores = qp_scorepos

global qp;
y = qp.i(1,1:qp.n);
I = find(y == 1);
w = qp.w + qp.w0.*qp.wreg;
scores = score(w,qp.x,I) / qp.Cpos;
end

% Computes expected number of nonzeros in sparse feature vector
function len = sparselen(model)
numblocks = 0;
feat = zeros(model.len,1);
for p = model.components
    if ~isempty(p.biasid)
        x = model.bias(p.biasid(1));    % use only one biase
        i1 = x.i;
        i2 = i1 + numel(x.w) - 1;
        feat(i1:i2) = 1;
        numblocks = numblocks + 1;
    end
    if ~isempty(p.appid)
        x  = model.apps(p.appid(1));  % use only one appearance filter
        i1 = x.i;
        i2 = i1 + numel(x.w) - 1;
        feat(i1:i2) = 1;
        numblocks = numblocks + 1;
    end
    assert(isempty(p.gauid) == (p.parent == 0), ...
        'gauid should be empty iff part is parent');
    if ~isempty(p.gauid)
        % use only one kind of deformation in each mixture
        assert(numel(p.gauid) == 1, ...
            'Need one deformation Gaussian per non-root part');
        x  = model.gaus(p.gauid(1));
        i1 = x.i;
        i2 = i1 + numel(x.w) - 1;
        feat(i1:i2) = 1;
        numblocks = numblocks + 1;
    end
end

% Number of entries needed to encode a block-sparse representation
%   1 + numberofblocks*2 + #nonzeronumbers
len = 1 + numblocks*2 + sum(feat);
end