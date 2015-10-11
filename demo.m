% Self-contained demo script. After you download FLIC, you can run this,
% sit back and watch the blinkenlights :-)

function demo

startup;
conf = get_conf;
[flic_data, pairs] = get_flic(conf.dataset_dir, conf.cache_dir);

parfor i=1:size(pairs, 1)
    fst_idx = pairs(i, 1);
    snd_idx = pairs(i, 2);
    fst = flic_data(fst_idx);
    snd = flic_data(snd_idx);
    fprintf('Computing flow for %s->%s\n', name(fst), name(snd));
    flow = imflow(fst.image_path, snd.image_path);
end

function n = name(datum)
[~,n,~] = fileparts(datum.image_path);
