function headless_detection_vis(dataset, pose_dets, dest_dir, pa)
%HEADLESS_DETECTION_VIS Visualise dets like headless_ds_vis does for GTs
% Nice for making pretty movies
figure('Visible', 'off');
axes('Visible', 'off');
set(0, 'DefaulttextInterpreter', 'none')
mkdir_p(dest_dir);
all_data = dataset.data;
seqs = dataset.seqs;

if exist('pa', 'var')
    fprintf('Plotting limbs\n');
    plotter = @(p) plot_limbs(p, pa);
else
    fprintf('Plotting joints\n');
    plotter = @plot_joints;
end

parfor seq_idx=1:length(dataset.seqs)
    seq = seqs{seq_idx};
    
    fprintf('Working on seq %i/%i\n', seq_idx, length(seqs)); %#ok<PFBNS>
    for frame_idx=1:length(seq)
        datum = all_data(seq(frame_idx)); %#ok<PFBNS>
        pred_joints = pose_dets{seq_idx}{frame_idx};
        
        im = readim(datum);
        imagesc(im);
        axis image off;
        axis equal;
        
        hold on;
        plotter(pred_joints);
        hold off;
        
        label = get_label(datum, seq(frame_idx));
        text(-50, -50, label, 'Interpreter', 'none');
        result_path = fullfile(dest_dir, ...
            sprintf('seq-%03i-frame-%04i.jpg', seq_idx, frame_idx));
        print(gcf, '-djpeg', result_path, '-r 300');
    end
end
end

function label = get_label(datum, idx)
if hasfield(datum, 'image_path')
    label = datum.image_path;
else
    label = sprintf('Index %i', idx);
end
end
