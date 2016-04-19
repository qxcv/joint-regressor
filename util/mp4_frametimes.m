function times = mp4_frametimes(mp4_path)
%MP4_FRAMETIMES Return time offsets for each frame in an MP4 file
reader = VideoReader(mp4_path);
frame_time = 1/reader.FrameRate;
% Video frame range is [0, reader.Duration). Of course, the Matlab docs
% don't *say* that the range is exclusive on one side, because why bother
% being clear on minor issues like whether the last frame of your video
% even exists?
end_time = reader.Duration - frame_time / 2;
times = 0:frame_time:end_time;
end
