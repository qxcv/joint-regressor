% HACKY HACKY PLAYING WITH THINGS WOOOOOOO

startup;
conf = get_conf_mpii;
[mpii_data, train_pairs, val_pairs] = get_mpii_cooking(conf.dataset_dir, ...
                                                       conf.cache_dir);
% Skippety                                                
mpii_data = mpii_data(1:3:end);

% Look at a bunch of pairs at a time
grid_size = 4;
for start_idx=1:grid_size^2:length(mpii_data)
    fprintf('Loading\n');
    data = mpii_data(start_idx:start_idx+grid_size^2);
    hold off;
    for row=1:grid_size
        for col=1:grid_size
            index = (row-1)*grid_size + col;
            if index > length(mpii_data)
                break;
            end
            subplot(grid_size, grid_size, index);
            imshow(readim(data(index)));
            axis equal;
            text(75, 300, num2str(data(index).scene_num), 'Color', 'red');
            % see http://au.mathworks.com/matlabcentral/newsreader/view_thread/149202
            sub_pos = get(gca,'position'); % get subplot axis position
            set(gca,'position',sub_pos.*[1 1 1.3 1.3]); % stretch its width and height
        end
    end
    if 1 == waitforbuttonpress
        % exit on key press
        break;
    end
end