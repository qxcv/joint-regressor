function [w,wreg,w0,noneg] = model2vec(model)
% [w,wreg,w0,nonneg] = model2vec(model)
w     = zeros(model.len,1);
w0    = zeros(model.len,1);
wreg  = ones(model.len,1);
noneg = uint32([]);

for x = model.bias
  j = x.i:x.i+numel(x.w)-1;
  w(j) = x.w;
end

for x = model.apps
  j = x.i:x.i+numel(x.w)-1;
  w(j) = x.w;
  % w0(j) is a lower bound for weight j (appearance weight)
  %w0(j) = 1e-3; % was 0.001 in CY code
  noneg = [noneg uint32(j)]; %#ok<AGROW>
end

for x = model.gaus
  j = x.i:x.i+numel(x.w)-1;
  w(j) = x.w;
  % Enforce minimum quadratic deformation cost
  j = [j(1),j(3)];
  % We actually need a minimum for these terms, because otherwise we will
  % get degenerate deformation parabolas which cause the distance transform
  % to crash (it divides by the weights of the squared terms).
  %w0(j) = 1e-3; % was 0.001 in CY code
  w0(j) = 1e-10;
  noneg = [noneg uint32(j)]; %#ok<AGROW>
end

% Regularize root biases differently
b = model.components(model.root).biasid;
x = model.bias(b);
j = x.i:x.i+numel(x.w)-1;
wreg(j) = .001;
