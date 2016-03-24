function num_collected = force_pygc()
%FORCE_PYGC Force Matlab Python interpreter to collect garbage.
% It seems like Maltab doesn't do this properly, meaning that long-running
% Matlab jobs which use the py.* interface can leak memory badly (I found
% this was happening to GPU memory when I was using Keras).
gc = py.importlib.import_module('gc');
num_collected = gc.collect.feval();
if num_collected > 0
    fprintf('Python collected %i garbage objects\n', num_collected);
end
end