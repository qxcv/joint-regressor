% Test that the third-party semaphore library I'm using is adequate to
% protect file access in parfor loops.

function test_semaphore_fileops()

filename = tempname;
iterations = 1e5;

fprintf('Without a semaphore\n');
init_file(filename);
fprintf('Initial value: %i\n', read_result(filename));
parfor i=1:iterations
    increment(filename);
end
fprintf('Result: %i\n', read_result(filename));

fprintf('With a semaphore\n');
init_file(filename);
fprintf('Initial value: %i\n', read_result(filename));
key = randi(2^15-1);
semaphore('create', key, 1);
parfor i=1:iterations
    semaphore('wait', key);
    increment(filename);
    semaphore('post', key);
end
semaphore('destory', key);
fprintf('Result: %i\n', read_result(filename));
end

function init_file(filename)
fp = fopen(filename, 'w');
fprintf(fp, '0');
fclose(fp);
end

function result = read_result(filename)
fp = fopen(filename, 'r');
result = fscanf(fp, '%i');
fclose(fp);
end

function increment(filename)
value = read_result(filename);
fp = fopen(filename, 'w');
fprintf(fp, '%i', value+1);
fclose(fp);
end