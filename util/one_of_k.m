function rv = one_of_k(classes, num_classes)
%ONE_OF_K Produce one-of-K vectors (as rows) for each datum in the given
% vector classes, assuming num_classes classes in total.
data_size = numel(classes);
rv = zeros(data_size, num_classes);
data_indices = sub2ind(size(rv), 1:data_size, classes(:)');
rv(data_indices) = 1;
end