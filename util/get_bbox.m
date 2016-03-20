function bbox = get_bbox(coords)
%GET_BBOX Get bounding box for rows of (x, y) coordinates
assert(size(coords, 2) == 2 && ismatrix(coords));
maxes = max(coords, [], 1);
mins = min(coords, [], 1);
xmin = mins(1);
ymin = mins(2);
sizes = maxes - mins + 1;
width = sizes(1);
height = sizes(2);
bbox = [xmin ymin width height];
end