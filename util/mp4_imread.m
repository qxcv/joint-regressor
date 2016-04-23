function frame = mp4_imread(mp4_path, frame_time)
%MP4_IMREAD Read single frame from MP4

assert(frame_time >= 0 ...
    && (isa(frame_time, 'single') || isa(frame_time, 'double')));

% VideoReader expects a double for times
ft_double = double(frame_time);
reader = VideoReader(mp4_path, 'CurrentTime', ft_double);
assert(reader.hasFrame(), 'Time %f out of range for %s', ft_double, mp4_path);
frame = reader.readFrame();
end

