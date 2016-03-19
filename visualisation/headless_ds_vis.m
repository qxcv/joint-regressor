function headless_ds_vis(dataset, dest_dir)
%HEADLESS_DS_VIS Visualise whole dataset and plots to directory.
num_data = length(dataset.data);
figure('Visible', 'off');
assert(~system(['mkdir -p ' dest_dir]));
for i=1:num_data
    if mod(i, 100) == 0
        fprintf('Done %i/%i images\n', i, num_data);
    end
    datum = dataset.data(i);
    show_datum(datum);
    result_path = fullfile(dest_dir, sprintf('%06i.jpg', i));
    print(gcf, '-djpeg', result_path, '-r 150');
end
end