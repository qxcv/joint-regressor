% Cache various statistics from the model data structure for later use
function [components, apps] = modelcomponents(model)
cell_components = cell(1, 1);

for subpose_idx = 1:length(model.components)
    subpose = model.components(subpose_idx);
    % store the scale of each part relative to the component root
    parent_idx = subpose.parent;
    assert(parent_idx < subpose_idx);
    subpose.b = [model.bias(subpose.biasid).w];
    subpose.b = reshape(subpose.b, [1 size(subpose.biasid)]);
    subpose.biasI = [model.bias(subpose.biasid).i];
    subpose.biasI = reshape(subpose.biasI, size(subpose.biasid));
    
    x = model.apps(subpose.appid);
    
    subpose.appI = x.i;
    
    % Remember that means are already in subpose.subpose_disps (see
    % build_model for more).
    if subpose_idx ~= model.root
        gauid = subpose.gauid;
        assert(isscalar(gauid));
        gau = model.gaus(gauid);
        subpose.gauw = gau.w;
        subpose.gauI = gau.i;
    else
        subpose.gauw = [];
        subpose.gauI = [];
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
