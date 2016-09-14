function set_pyversion()
%SET_PYVERSION Ensure we're using Python 2.7
[version_string, exec_path, py_loaded] = pyversion;
if ~strcmp(version_string, '2.7')
    if py_loaded
        warning('set_pyversion:WrongVersion', ...
            ['Wrong version of Python loaded (v%s, path %s)! Try to ' ...
             'restart Matlab and re-run set_pyversion.'], ...
             version_string, exec_path);
        return;
    end
    % For some reason, `pyversion 2.7` doesn't work on paloalto
    [stat, out] = system('which python2.7');
    if stat ~= 0
        warning('set_pyversion:WrongVersion', ...
            'Couldn''t get Python 2.7 path (`which` status %d)', stat);
        return;
    end
    true_exe = strtrim(out);
    pyversion(true_exe);
end
end

