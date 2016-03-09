function test_im_flow_resize
%TEST_IM_FLOW_RESIZE Visual test of im_flow_resize
FRAME_DIR = './datasets/mpii-cooking-pose-challenge/data/train_data/images/';
FIRST_FRAME = 'img_000000.jpg';
SECOND_FRAME = 'img_000001.jpg';

paths = fullfile(FRAME_DIR, {FIRST_FRAME, SECOND_FRAME});
frame1 = imread(paths{1});
frame2 = imread(paths{2});
flow = imflow(paths{1}, paths{2});

% Show originals
figure('Name', 'Flow scale comparison');
subplot(2, 3, 1);
imshow(frame1);
subplot(2, 3, 2);
imshow(frame2);
subplot(2, 3, 3);
imshow(pretty_flow(flow));

% Now go to half scale
new_size = size(flow) / 4;
new_size = new_size(1:2);
frame1_small = imresize(frame1, new_size);
frame2_small = imresize(frame2, new_size);
flow_small = smart_resize_flow(flow, new_size);
true_small_flow = broxOpticalFlow(frame1_small, frame2_small);

% Show new versions
subplot(2, 3, 4);
imshow(frame1_small);
subplot(2, 3, 5);
imshow(frame2_small);
subplot(2, 3, 6);
imshow(pretty_flow(flow_small));

figure('Name', 'True small flow');
imshow(pretty_flow(true_small_flow));
end

