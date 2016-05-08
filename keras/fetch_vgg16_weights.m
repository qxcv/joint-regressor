function weight_path = fetch_vgg16_weights()
%FETCH_VGG16_WEIGHTS Grab ILSVRC weights for fine-tuning
% Downloads weights if they haven't been downloaded and returns path to
% weights

% Where do we store the weights?
weight_fn = 'vgg16_weights.h5';
this_dir = fileparts(mfilename('fullpath'));
weight_dir = fullfile(this_dir, 'vggnet');
assert(~~exist(weight_dir, 'dir'), [weight_dir ' should exist already ' ...
    '(it should be in the downloaded code)']);
weight_path = fullfile(weight_dir, weight_fn);

if exist(weight_path, 'file')
    fprintf('Weights already exist at "%s", skipping download\n', ...
        weight_path);
else
    % Grab the URL from a file and download it
    url_list_path = fullfile(weight_dir, 'download_urls.txt');
    fp = fopen(url_list_path);
    weight_url = [];
    while isempty(weight_url)
        line = fgetl(fp);
        assert(isa(line, 'char'), 'Couldn''t read URL from "%s"', ...
            url_list_path);
        if ~all(isspace(line)) && line(1) ~= '#'
            weight_url = line;
        end
    end
    fprintf('Downloading weights from "%s" to "%s"\n', weight_url, ...
        weight_path);
    websave(weight_path, weight_url);
end
end
