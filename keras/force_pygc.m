function num_collected = force_pygc()
%FORCE_PYGC Force Matlab Python interpreter to collect garbage.
% It seems like Maltab doesn't do this properly, meaning that long-running
% Matlab jobs which use the py.* interface can leak memory badly (I found
% this was happening to GPU memory when I was using Keras).
gc = py.importlib.import_module('gc');
% gc.collect.feval() returns an integer. For some reason, Matlab only
% converts that to a Matlab-native integer (as opposed to a py.int)
% *sometimes*. The int64() wrap should work fine.
num_collected = int64(gc.collect.feval());
if num_collected > 0
    fprintf('Python collected %i garbage objects\n', num_collected);
end
end
