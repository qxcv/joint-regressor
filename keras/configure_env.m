function configure_env
%CONFIGURE_ENV Activate virtualenv for Keras, initialising if necessary

force_pygc();

persistent configured;
if isempty(configured)
    configured = true;
else
    assert(configured);
    return
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
weak_setenv('HDF5_DISABLE_VERSION_CHECK', '1');
weak_setenv('THEANO_FLAGS', 'floatX=float32,device=gpu0');
% Activate the environment
at_path = fullfile(pwd, 'keras/env/bin/activate_this.py');
arg_dict = py.dict({{'__file__', at_path}});
py.execfile(at_path, arg_dict);
end

function weak_setenv(name, value)
% Only setenv(name, value) if name() empty or not set
if isempty(getenv(name))
    setenv(name, value)
else
    fprintf('env var %s non-empty, so configure_env won''t touch it\n', name);
end
end