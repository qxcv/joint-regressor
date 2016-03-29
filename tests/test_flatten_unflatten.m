function test_flatten_unflatten
%TEST_FLATTEN_UNFLATTEN Random checks to make sure that flatten is the
%inverse of unflatten
for i=1:1000
    if mod(i, 500) == 0
        fprintf('Flatten -> unflatten iter %i\n', i);
    end
    num_coords = 2 * randi([1 50]);
    coords = rand([num_coords 2]);
    flat = flatten_coords(coords);
    new_coords = unflatten_coords(flat);
    assert(every(new_coords == coords));
end

for i=1:1000
    if mod(i, 500) == 0
        fprintf('Unflatten -> flatten iter %i\n', i);
    end
    vec_len = 2 * randi([1 50]);
    flat = rand([vec_len 1]);
    coords = unflatten_coords(flat);
    coords_t = unflatten_coords(flat');
    assert(every(coords == coords_t));
    new_flat = flatten_coords(coords);
    assert(every(new_flat == flat));
end

fprintf('Tests complete, everything works\n');
end

function rv = every(bools)
% Flattening version of all()
rv = all(bools(:));
end