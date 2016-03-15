% Cache various statistics from the model data structure for later use
function [components, apps] = modelcomponents(model)
cell_components = cell(1, 1);

for k = 1:length(model.components)
    p = model.components(k); % has nbh_IDs
    nbh_N = numel(p.nbh_IDs);
    p.Im = cell(nbh_N, 1);
    % store the scale of each part relative to the component root
    par = p.parent;
    assert(par < k);
    p.b = [model.bias(p.biasid).w];
    p.b = reshape(p.b, [1 size(p.biasid)]);
    p.biasI = [model.bias(p.biasid).i];
    p.biasI = reshape(p.biasI, size(p.biasid));
    
    x = model.apps(p.appid);
    
    p.sizy = model.tsize(1);
    p.sizx = model.tsize(2);
    p.appI = x.i;
    
    for d = 1:nbh_N
        for m = 1:numel(p.gauid{d})
            x = model.gaus(p.gauid{d}(m));
            p.gauw{d}(m,:)  = x.w;
            p.gauI{d}(m) = x.i;
            mean_x = x.mean(1);
            mean_y = x.mean(2);
            
            p.mean_x{d}(m) = mean_x;
            p.mean_y{d}(m) = mean_y;
        end
    end
    cell_components{1}(k) = p;
end
apps = cell(length(model.apps), 1);

for i = 1:length(apps)
    apps{i} = model.apps(i).w;
end

% Unwrap cell array; I can only assume a cell array was used to make struct
% access easier (apparently Matlab doesn't require you to declare a struct
% ahead of time if it's a cell? WTF?)
components = cell_components{1};