function headless_pair_vis(dataset, dest_dir)
%HEADLESS_PAIR_VIS Like headless_ds_vis, but plots pairs instead of frames
figure('Visible', 'off');
axes('Visible', 'off');
mkdir_p(dest_dir);
all_data = dataset.data;
num_pairs = dataset.num_pairs;
all_pairs = dataset.pairs;
parfor pair_idx=1:num_pairs
    if mod(pair_idx, 100) == 0
        fprintf('Done %i/%i pairs\n', pair_idx, num_pairs);
    end
    
    pair = all_pairs(pair_idx);
    pair_data = all_data([pair.fst, pair.snd]); %#ok<PFBNS>
    
    for sub_idx=1:2
        subplot(2, 1, sub_idx);
        datum = pair_data(sub_idx);
        show_datum(pair_data(sub_idx));
        label = get_label(datum, pair_idx);
        text(-50, -50, label, 'Interpreter', 'none');
    end
    
    result_path = fullfile(dest_dir, sprintf('%06i.jpg', pair_idx));
    print(gcf, '-djpeg', result_path, '-r 150');
end
end

function label = get_label(datum, idx)
if hasfield(datum, 'image_path')
    label = datum.image_path;
else
    label = sprintf('Index %i', idx);
end
end
