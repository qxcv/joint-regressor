function [score,Ix,Iy,Imc,Imp] = passmsg(child, parent, cbid, pbid)
assert(numel(cbid) == 1 && numel(pbid) == 1);
Ny  = size(parent.score,1);
Nx  = size(parent.score,2);

MC = numel(child.gauid{cbid});
MP = numel(parent.gauid{pbid});

[score0,Ix0,Iy0] = deal(zeros(Ny,Nx,MC,MP));
for mc = 1:MC
  for mp = 1:MP
    % XXX: .defMap and .pdw no longer exist. They were just for the "prior
    % of deformation", which is the limb type which Chen & Yuille's CNN
    % predicted. I'm not predicting limb types, so I don't need that.
    [score0(:,:,mc,mp), Ix0(:,:,mc,mp), Iy0(:,:,mc,mp)] = ...
      distance_transform( double(child.score + (child.pdw(cbid)*child.defMap{cbid}(:,:,mc))), ...
      child.gauw{cbid}(mc,:), parent.gauw{pbid}(mp,:), ...
      [child.mean_x{cbid}(mc), child.mean_y{cbid}(mc)], ...
      ... XXX: I've ripped the variance thing out of here, but I should rip
      ... it out of distance_transform as well.
      [1 1], ... [child.var_x{cbid}(mc), child.var_y{cbid}(mc)], ...
      [parent.mean_x{pbid}(mp), parent.mean_y{pbid}(mp)], ...
      [1 1], ... [parent.var_x{pbid}(mp), parent.var_y{pbid}(mp)], ...
      int32(Nx), int32(Ny) );
    
    score0(:,:,mc,mp) = score0(:,:,mc,mp) + parent.pdw(pbid)*parent.defMap{pbid}(:,:,mp);
  end
end
score = reshape(score0, size(score0, 1),size(score0, 2), MC*MP);
[score, Imcp] = max(score, [], 3);
[Imc, Imp] = ind2sub([MC,MP], Imcp);
[Ix, Iy] = deal(zeros(Ny,Nx));
for i = 1:Ny
  for j = 1:Nx
    Ix(i,j) = Ix0(i,j,Imc(i,j),Imp(i,j));
    Iy(i,j) = Iy0(i,j,Imc(i,j),Imp(i,j));
  end
end
