function frame = mp4_imread(mp4_path, frame_time)
%MP4_IMREAD Read single frame from MP4
assert(frame_time >= 0);
reader = VideoReader(mp4_path, 'CurrentTime', frame_time);
assert(reader.hasFrame(), 'Time %f out of range for %s', frame_time, mp4_path);
frame = reader.readFrame();
end

