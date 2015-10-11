% General startup code

addpath ./datasets/;
addpath ./util/;

conf = get_conf();
mkdir(conf.cache_dir);
addpath(conf.ext_dir);
get_deps(conf.ext_dir, conf.cache_dir);