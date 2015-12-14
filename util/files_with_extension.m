function paths = files_with_extension(directory, extension)
%FILES_WITH_EXTENSION List files in dir with given extension
dir_result = dir(fullfile(directory, ['*' extension]));
paths = cell(1, length(dir_result));
for i=1:length(dir_result)
    paths{i} = fullfile(directory, dir_result(i).name);
end
end