% General startup code

if isunix && usejava('jvm') && isempty(gcp('nocreate'))
    [stat, res] = system('free -g | sed -e ''s/\s\+/ /g'' | grep ^Mem: | cut -f 2 -d " "');
    if ~stat && str2double(res) > 32
        % If we have memory to burn, then we can make a pool that lasts a
        % few days.
        fprintf('Starting long-lived pool\n');
        parpool('IdleTimeout', 48 * 60);
    end
    clear stat res;
end

addpath_full ./;
addpath_full ./eval;
addpath_full ./datasets/;
addpath_full ./detect/;
addpath_full ./util/;
addpath_full ./train/;
addpath_full ./visualisation/;
addpath_full ./tests/;
addpath_full ./keras/;
addpath_full ./cy/;

% Chen & Yuille code

old = cd('./cy');
cd_reset = onCleanup(@() cd(old));
CY_startup;
CY_compile;
cd_reset.task();

% Cherian et al. code
cd('./cmas');
CMAS_startup;
cd_reset.task();

conf = get_conf();
if ~exist(conf.cache_dir, 'dir')
    % We have this stupid guard to avoid Matlab warnings
    mkdir(conf.cache_dir);
end
addpath_full(conf.ext_dir);
get_deps;

clear old cd_reset conf;
