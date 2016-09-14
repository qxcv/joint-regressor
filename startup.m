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
% Things used at training- and test-time
addpath_full ./common/;
% Chen & Yuille code
addpath_full ./cy/;
% Dataset loading code
addpath_full ./datasets/;
% Things used for test-time detection
addpath_full ./detect/;
% Things used or statistical evaluation
addpath_full ./eval;
% CNN-related code (mostly Python)
addpath_full ./keras/;
% Small selection of (unit-style) tests
addpath_full ./tests/;
% Code for training CNN, SSVM, etc.
addpath_full ./train/;
% Utilities not specific to this project (e.g. Matlab helpers)
addpath_full ./util/;
% Visualisation code
addpath_full ./visualisation/;

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

set_pyversion;

conf = get_conf();
% if ~exist(conf.cache_dir, 'dir')
%     % We have this stupid guard to avoid Matlab warnings
%     mkdir(conf.cache_dir);
% end
addpath_full(conf.ext_dir);
get_deps;

clear old cd_reset conf;
