% Downloads FLIC data, if necessary

function get_flic(dest_dir, cache_dir)
FLIC_URL = 'http://vision.grasp.upenn.edu/video/FLIC.zip';
DEST_PATH = fullfile(dest_dir, 'FLIC/');
CACHE_PATH = fullfile(cache_dir, 'FLIC.zip');

if ~exist(DEST_PATH, 'dir')
    if ~exist(CACHE_PATH, 'file')
        fprintf('Downloading FLIC from %s\n', FLIC_URL);
        websave(CACHE_PATH, FLIC_URL);
    end
    fprintf('Extracting FLIC data to %s\n', DEST_PATH);
    unzip(CACHE_PATH, dest_dir);
end;