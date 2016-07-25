function visualise_rescored_order(out_dir)
%VISUALISE_RESCORED_ORDER Visualisation of rescored poses (in descending
%order of score).
assert(nargin >= 1);

startup;
conf = get_conf_mpii;

% Visualisation junk
joint_pa = [0 1 2:5 1 7:10];

cy_pred_path = 'topk/cy-topk-mpii-dets.mat';
[~, test_seqs] = ...
    parload(cy_pred_path, 'results', 'mpii_test_seqs');
test_seqs = convert_test_seqs(test_seqs);
num_seqs = length(test_seqs.seqs);
pair_cache_dir = fullfile(conf.cache_dir, 'cy-rescore-saved-pairs');
mkdir_p(out_dir);

parfor seq_idx=1:num_seqs
    seq = test_seqs.seqs{seq_idx}; %#ok<PFBNS>
    num_pairs = length(seq) - 1;
    for fst_idx=1:num_pairs
        fprintf('Pair %i/%i (seq %i/%i)\n', fst_idx, num_pairs, ...
            seq_idx, num_seqs);
        cache_fn = sprintf('seq-%i-pair-%i.mat', fst_idx, seq_idx);
        cache_path = fullfile(pair_cache_dir, cache_fn);
        [rscores, recovered] = parload(cache_path, 'rscores', 'recovered');
        d1 = test_seqs.data(seq(fst_idx));
        d2 = test_seqs.data(seq(fst_idx+1));
        im1 = readim(d1);
        im2 = readim(d2);
        
        % Now need to visualise everything in a 2 * length(rscores) grid
        ncandidates = length(rscores);
        for pred_idx=1:ncandidates
            joints = recovered{pred_idx};
            j1 = joints{1};
            j2 = joints{2};
            
            % First frame
            save_candidate(out_dir, im1, j1, joint_pa, seq_idx, fst_idx, ...
                pred_idx, 1, sprintf('Pred %i', pred_idx));
            
            % Second frame
            save_candidate(out_dir, im2, j2, joint_pa, seq_idx, fst_idx, ...
                pred_idx, 2, sprintf('rscore=%f', rscores(pred_idx)));
        end
        render_to_html(out_dir, seq_idx, fst_idx, ncandidates)
    end
end
end

% Cuttin' and pastin' from rescoring_cy_mpii
% This makes me a little sad :(
function test_seqs = convert_test_seqs(test_seqs)
for i=1:length(test_seqs.data)
    old_path = test_seqs.data(i).image_path;
    new_path = regexprep(old_path, '^./dataset/mpii/', './datasets/');
    assert(~~exist(new_path, 'file'), 'Image %s is missing', new_path);
    test_seqs.data(i).image_path = new_path;
end
end

function save_candidate(dest_dir, im, joints, joint_pa, seq_num, pair_num, cn, fn, extra_text)
real_dest = fullfile(dest_dir, sprintf('.images-%i-%i', seq_num, pair_num));
mkdir_p(real_dest);
dest_path = fullfile(real_dest, sprintf('%i-%i.jpg', cn, fn));
figure();
plot_width = size(im, 2);
plot_height = size(im, 1);
text_pos = num2cell([plot_width, plot_height] .* 0.1);
imagesc(im);
axis('image', 'off');
axis('equal');
hold on;
plot_limbs(joints, joint_pa);
text(text_pos{1:2}, extra_text, 'Color', 'green', 'BackgroundColor', 'black');
hold off;
tightfig;
print(gcf, '-djpeg', dest_path, '-r100');
close(gcf);
end

function render_to_html(dest_dir, seq_num, pair_num, candidates)
% Yes, rendering HTML eventually became the faster option, once I realised
% Matlab's subplot() is useless if you want your images to be visible.
html_template = [...
    '<!DOCTYPE html><html><head><meta charset="utf-8">' ...
    '<title>Visualisation of seq %i, pair %i</title>' ...
    '</head><body>' ...
    '<h1>Visualisation of seq %i pair %i</h1>' ...
    '<table><tr><th>Frame 1</th><th>Frame 2</th></tr>%s</table>' ...
    '</body></html>'];
render_html = @(sn, pn, data) sprintf(html_template, sn, pn, sn, pn, data);
rows = '';
image_dir = sprintf('.images-%i-%i', seq_num, pair_num);
im_tag = @(cn, fn) sprintf('<img src="%s/%i-%i.jpg"/>', image_dir, cn, fn);
for i=1:candidates
    img1 = im_tag(i, 1);
    img2 = im_tag(i, 2);
    rows = [rows '<tr><td>' img1 '</td><td>' img2 '</td></tr>']; %#ok<AGROW>
end
rendered = render_html(seq_num, pair_num, rows);
mkdir_p(dest_dir);
fd = fopen(fullfile(dest_dir, ...
    sprintf('candidates-seq-%03i-pair-%02i.html', seq_num, pair_num)), ...
    'w');
fprintf(fd, '%s', rendered);
fclose(fd);
end
