function convnet = get_net(conf)
%GET_NET Get a Caffe net for testing
if conf.cnn.gpu_id >= 0
    caffe.set_mode_gpu();
    caffe.set_device(conf.cnn.gpu_id);
else
    caffe.set_mode_cpu();
end
convnet = caffe.Net(conf.cnn.deploy_prototxt, conf.cnn.model, 'test');
end

