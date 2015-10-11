% Self-contained demo script. After you download FLIC, you can run this,
% sit back and watch the blinkenlights :-)

startup;
conf = get_conf;
get_flic(conf.dataset_dir, conf.cache_dir);