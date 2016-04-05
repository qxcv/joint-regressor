% General startup code

addpath_full ./;
addpath_full ./datasets/;
addpath_full ./detect/;
addpath_full ./util/;
addpath_full ./train/;
addpath_full ./visualisation/;
addpath_full ./tests/;
addpath_full ./keras/;
addpath_full ./cy/;
old = cd('./cy');
CY_startup;
CY_compile;
cd(old);

conf = get_conf();
if ~exist(conf.cache_dir, 'dir')
    % We have this stupid guard to avoid Matlab warnings
    mkdir(conf.cache_dir);
end
addpath_full(conf.ext_dir);
get_deps;
