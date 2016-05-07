function cells = map2cell(input_map)
%MAP2CELL Convert a k: v map to a {k1, v1, k2, v2} cell array.
assert(isa(input_map, 'containers.Map'));
cells = cell([1, input_map.Count * 2]);
key_names = input_map.keys;
for key_idx=1:length(key_names)
    base = 2 * key_idx - 1;
    key = key_names{key_idx};
    cells{base} = key;
    cells{base+1} = input_map(key);
end
end

