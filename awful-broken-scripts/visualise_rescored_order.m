function visualise_rescored_order(out_dir)
%VISUALISE_RESCORED_ORDER Visualisation of rescored poses (in descending
%order of score).
assert(nargin >= 1);

startup;
conf = get_conf_mpii;

% Visualisation junk
joint_pa = [0 1 2:5 1 7:10];
figure('Visible', 'off');
axes('Visible', 'off');

cy_pred_path = 'topk/cy-topk-mpii-dets.mat';
[~, test_seqs] = ...
    parload(cy_pred_path, 'results', 'mpii_test_seqs');
test_seqs = convert_test_seqs(test_seqs);
num_seqs = length(test_seqs.seqs);
pair_cache_dir = fullfile(conf.cache_dir, 'cy-rescore-saved-pairs');
mkdir_p(out_dir);

for seq_idx=1:num_seqs
    seq = test_seqs.seqs{seq_idx};
    num_pairs = length(seq) - 1;
    for fst_idx=1:num_pairs
        fprintf('Pair %i/%i (seq %i/%i)\n', fst_idx, num_pairs, ...
            seq_idx, num_seqs);
        cache_fn = sprintf('seq-%i-pair-%i.mat', fst_idx, seq_idx);
        cache_path = fullfile(pair_cache_dir, cache_fn);
        [rscores, recovered] = parload(cache_path, 'rscores', 'recovered');
        d1 = test_seqs.data(seq(fst_idx)); %#ok<PFBNS>
        d2 = test_seqs.data(seq(fst_idx+1)); %#ok<PFBNS>
        im1 = readim(d1);
        im2 = readim(d2);
        plot_width = size(im1, 2);
        plot_height = size(im1, 1);
        text_pos = num2cell([plot_width, plot_height] .* 0.1);
        stp = @(m, n, p) subtightplot(m, n, p, 0, 0, 0);
        
        % Now need to visualise everything in a 2 * length(rscores) grid
        nrows = min([10 length(rscores)]);
        fig = figure();
        hold on;
        for pred_idx=1:nrows
            start_p = pred_idx * 2 - 1;
            joints = recovered{pred_idx};
            j1 = joints{1};
            j2 = joints{2};
            
            % First frame
            stp(nrows, 2, start_p);
            imagesc(im1);
            axis('image', 'off');
            axis('equal');
            plot_limbs(j1, joint_pa);
            text(text_pos{1:2}, sprintf('Pred %i', pred_idx), 'Color', 'green');
            
            % Second frame
            stp(nrows, 2, start_p+1);
            imagesc(im2);
            axis('image', 'off');
            axis('equal');
            plot_limbs(j2, joint_pa);
            text(text_pos{1:2}, sprintf('rscore=%f', rscores(pred_idx)), 'Color', 'green');
        end
        hold off;
        
        result_path = fullfile(out_dir, sprintf('seq-%i-pair-%i.jpg', ...
            seq_idx, fst_idx));
        
        % dpi is arbitrary, since we control figure size anyway
        dpi = 100;
        resize_factor = 0.25;
        pixel_height = size(im1, 1) * nrows;
        current_pos = get(fig, 'Position');
        old_width = current_pos(3);
        old_height = current_pos(4);
        pixel_width = pixel_height * old_width / old_height;
        pixel_size = [pixel_width, pixel_height];
        printed_size = pixel_size .* resize_factor ./ dpi;
        set(fig, 'Units', 'Inches', ...
            'Position', [0, 0, printed_size], ...
            'PaperUnits', 'Inches', ...
            'PaperPosition', [0, 0, printed_size]);
        % set(gcf,'PaperPositionMode','auto')
        print(fig, '-djpeg', result_path, sprintf('-r%i', dpi));
        
        close(fig);
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
