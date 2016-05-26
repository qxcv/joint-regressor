function cfg = vis_configs
%VIS_CONFIGS Configurations for visualisation of original datasets.

% Cooking Activities
cfg.mpii_pa = nan([1 12]);
cfg.mpii_pa(3:8) = [3 3 3 4 5 6];

% Human3.6M
cfg.h36m_pa = nan([1 32]);
cfg.h36m_pa([26:28 18:20]) = [26 26 27 26 18 19];

% Poses in the Wild
cfg.piw_pa = nan([1 8]);
cfg.piw_pa([5:7 2:4]) = [5 5 6 5 2 3];
end
