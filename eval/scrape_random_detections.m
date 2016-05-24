function outputs = scrape_random_detections(num_to_fetch)
%SCRAPE_RANDOM_DETECTIONS Run some random training samples through the CNN
%   Stores relevant data in outputs(i).stack and outputs(i).labels. This is
%   a really hacky script which I use for debugging :)
conf = get_conf();
train_dir = fullfile(conf.cache_dir, 'val-patches');
h5_names = dir(fullfile(train_dir, '*.h5'));
all_h5s = fullfile(train_dir, {h5_names.name});
my_h5_path = all_h5s{randi(length(all_h5s))};
fprintf('Reading %s\n', my_h5_path);
stacks = h5read(my_h5_path, '/data');
labels = h5read(my_h5_path, '/label');
num_samples = size(stacks, 4);
rperm = randperm(num_samples);
chosen = rperm(1:num_to_fetch);
net = get_net(conf);
for i=1:length(chosen)
    j = chosen(i);
    outputs(i).stack = stacks(:, :, :, j);
    outputs(i).labels = detect_single(outputs(i).stack, net);
    outputs(i).true_labels = labels(:, j);
end
end

