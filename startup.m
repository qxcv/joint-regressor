% General startup code

addpath ./datasets/;
addpath ./util/;
addpath ./train/;

conf = get_conf();
if ~exist(conf.cache_dir, 'dir')
    % We have this stupid guard to avoid Matlab warnings
    mkdir(conf.cache_dir);
end
addpath(conf.ext_dir);
get_deps(conf.ext_dir, conf.cache_dir);