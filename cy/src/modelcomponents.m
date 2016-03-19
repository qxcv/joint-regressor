% Cache various statistics from the model data structure for later use
function [components, apps] = modelcomponents(model)
cell_components = cell(1, 1);

assert(false, 'Need to fix modelcomponents');

for subpose_idx = 1:length(model.components)
    subpose = model.components(subpose_idx); % has nbh_IDs
    nbh_N = numel(subpose.nbh_IDs);
    subpose.Im = cell(nbh_N, 1);
    % store the scale of each part relative to the component root
    parent_idx = subpose.parent;
    assert(parent_idx < subpose_idx);
    subpose.b = [model.bias(subpose.biasid).w];
    subpose.b = reshape(subpose.b, [1 size(subpose.biasid)]);
    subpose.biasI = [model.bias(subpose.biasid).i];
    subpose.biasI = reshape(subpose.biasI, size(subpose.biasid));
    
    x = model.apps(subpose.appid);
    
    subpose.sizy = model.tsize(1);
    subpose.sizx = model.tsize(2);
    subpose.appI = x.i;
    
    for d = 1:nbh_N
        for m = 1:numel(subpose.gauid{d})
            x = model.gaus(subpose.gauid{d}(m));
            subpose.gauw{d}(m,:)  = x.w;
            subpose.gauI{d}(m) = x.i;
            mean_x = x.mean(1);
            mean_y = x.mean(2);
            
            subpose.mean_x{d}(m) = mean_x;
            subpose.mean_y{d}(m) = mean_y;
        end
    end
    cell_components{1}(subpose_idx) = subpose;
end
apps = cell(length(model.apps), 1);

for i = 1:length(apps)
    apps{i} = model.apps(i).w;
end

% Unwrap cell array; I can only assume a cell array was used to make struct
% access easier (apparently Matlab doesn't require you to declare a struct
% ahead of time if it's a cell? WTF?)
components = cell_components{1};
end