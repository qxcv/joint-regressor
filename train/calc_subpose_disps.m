function pairwise_means = calc_subpose_disps(centroids, subpose_graph, shared_parts)
%CALC_SUBPOSE_DISPS Displacements between subposes for each poselet Uses
%centroids calculated with K-means, and asks, for each pair of adjacent
%sub-poses s1 and s2, and each of their poselet classes p1 and p2
%(respectively), what the "distance" between the first and second subposes
%and the first and second poselets is. Note that this is the distance
%between the centers of the poselets, which is taken to be the middle of
%the 224x224 frame which they are cropped from.

% Return value will probably be an S*P*P*2 array, where rv(i, p1, p2, :)
% gives the amount that the child subpose has to be moved to match its
% parent, where the child has pose p1 and the parent has pose p2.

% **Interesting note:** so I just realised why Matlab users always put the
% axes they want to access at the same time first (e.g. making coords
% 2*S*P*P instead of S*P*P*2). It's because accessing thing(1, 2, 3, :)
% gives you a 1x1x1xN array, which you then have to sqeeze & transpose to
% use.
num_poselets = length(centroids{1});
rv_size = [length(subpose_graph), num_poselets, num_poselets, 2];
% Initialise return value to all nans so that I can figure out when I've
% messed up :)
pairwise_means = nan(rv_size);

for child=1:length(subpose_graph)
    parent = subpose_graph(child);
    if ~parent
        continue
    end
    
    child_centroids = centroids{child};
    parent_centroids = centroids{parent};
    child_shareds = shared_parts{child}{1};
    parent_shareds = shared_parts{child}{2};
    
    for child_poselet=1:length(centroids{child})
        % Joint coordinates associated with child poselet
        child_coords = unflatten_coords(child_centroids(child_poselet, :));
        child_end = average_shareds(child_coords, child_shareds);
        for parent_poselet=1:length(centroids{parent})
            % Joint coordinates associated with parent centroid
            parent_coords = unflatten_coords(parent_centroids(parent_poselet, :));
            parent_end = average_shareds(parent_coords, parent_shareds);
            % disp is the amount by which the child has to be moved to
            % match the parent
            disp = child_end - parent_end;
            assert(isvector(disp)); 
            pairwise_means(child, child_poselet, parent_poselet, :) = disp;
        end
    end
end
end

function rv = average_shareds(coords, shares)
assert(isvector(shares));
assert(ismatrix(coords));
assert(size(coords, 2) == 2);
rv = mean(coords(shares, :), 1);
assert(isvector(rv));
end