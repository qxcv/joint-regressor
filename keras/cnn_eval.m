function [left_locs, right_locs, head_locs, classes] = cnn_eval(model, images_data, flow_data)
%CNN_EVAL Evaluates our CNN by calling into Python
% Our code uses Keras, so evaluating directly in Matlab (or a mex library)
% won't work here.
data_dict = py.dict();
data_dict{'images'} = images_data;
data_dict{'flow'} = flow_data;
results = model.predict(data_dict);
left_locs = cnn_numpy_to_matlab(results{'left'});
right_locs = cnn_numpy_to_matlab(results{'right'});
head_locs = cnn_numpy_to_matlab(results{'head'});
classes = cnn_numpy_to_matlab(results{'class'});
end