function model = vec2model(w,model)
%VEC2MODEL Incorporate QP-solver-optimised weights back into model.
% I think that the QP solver just treats all the weights as one big vector,
% so after it does its thing we need to copy weights back into their
% respective (meaningfully named/divided) model fields.
w = double(w);

% Biases
for i = 1:length(model.bias)
  x = model.bias(i);
  s = size(x.w);
  j = x.i:x.i+prod(s)-1;
  model.bias(i).w = reshape(w(j),s);
end

% apps
for i = 1:length(model.apps)
  x = model.apps(i);
  s = size(x.w);
  j = x.i:x.i+prod(s)-1;
  model.apps(i).w = reshape(w(j),s);
end

% prior of deformation parameters
for i = 1:length(model.pdefs)
  x = model.pdefs(i);
  s = size(x.w);
  j = x.i:x.i+prod(s)-1;
  model.pdefs(i).w = reshape(w(j),s);
end

% gauss parameters
for i = 1:length(model.gaus)
  x = model.gaus(i);
  s = size(x.w);
  j = x.i:x.i+prod(s)-1;
  model.gaus(i).w = reshape(w(j),s);
end

% Debug
w2 = model2vec(model);
assert(isequal(w,w2));
