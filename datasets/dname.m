function n = dname(datum)
if hasfield(datum, 'image_path')
    [~,n,~] = fileparts(datum.image_path);
else
    n = sprintf('s-%i-a-%s-c-%i-v-%i-f-%i', datum.subject, ...
        urlencode(datum.action), datum.camera, datum.video_id, ...
        datum.frame_no);
end
end
