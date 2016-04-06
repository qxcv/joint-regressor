function configure_env(cnn_conf)
%CONFIGURE_ENV Activate virtualenv for Keras, initialising if necessary

force_pygc();

persistent configured;
if ~isempty(configured)
    return;
end

if ~exist('./keras/env', 'dir')
    fprintf('Installing virtualenv in keras/env\n');
    init_command = ['virtualenv --system-site-packages ./keras/env '...
        '&& source ./keras/env/bin/activate '...
        '&& pip install -r ./keras/requirements.txt'];
    rv = system(init_command, '-echo');
    assert(rv == 0, 'Python package install failed');
end
% Set some environment variables
% XXX: Should have some control over these from get_config.m. Particularly
% the device Theano uses.
py_weak_setenv('HDF5_DISABLE_VERSION_CHECK', '1');
THEANO_FLAGS = sprintf('floatX=float32,device=gpu%i,lib.cnmem=%f', ...
    cnn_conf.gpu, cnn_conf.cnmem);
py_weak_setenv('THEANO_FLAGS', THEANO_FLAGS);
fprintf('THEANO_FLAGS=%s\n', py_getenv('THEANO_FLAGS'));
% Activate the environment
at_path = fullfile(pwd, 'keras/env/bin/activate_this.py');
arg_dict = py.dict({{'__file__', at_path}});
py.execfile(at_path, arg_dict);

configured = true;
end

% We need to use both Python's getenv/setenv. For some reason changes using
% Matlab's getenv/setenv don't propagate to the embedded Python
% interpreter.

function py_weak_setenv(name, value)
% Only setenv(name, value) if name() empty or not set in Python. Will set
% in Python and Matlab.
os = py.importlib.import_module('os');
if ~os.environ.has_key(name)
    os.environ.update(py.dict({{name, value}}));
    setenv(name, value);
else
    fprintf('env var %s non-empty, so configure_env won''t touch it\n', name);
end
end

function result = py_getenv(name)
os = py.importlib.import_module('os');
py_result = os.environ.get(name, '');
result = py_result.char;
end