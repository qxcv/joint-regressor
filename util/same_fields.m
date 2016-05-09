function all_equal = same_fields(conf1, conf2)
%SAME_FIELDS Check that two configurations have the same fields and types
assert(isstruct(conf1) && isstruct(conf2), 'Inputs must be structs');
names1 = fieldnames(conf1);
names2 = fieldnames(conf2);
all_names = union(names1, names2);

% Check names
diff_names = setdiff(all_names, intersect(names1, names2));
if ~isempty(diff_names)
    fprintf('Structs differ on fields %s\n', strjoin(diff_names, ', '));
    all_equal = false;
    return
end

% Check types
for name_idx=1:length(all_names)
    name = all_names{name_idx};
    class1 = class(conf1.(name));
    class2 = class(conf2.(name));
    if ~strcmp(class1, class2)
        fprintf('Field types differ for %s (%s vs. %s)\n', ...
            name, class1, class2);
        all_equal = false;
        return
    end
end

all_equal = true;
end

