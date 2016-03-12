function [nbh_IDs, global_IDs, target_IDs] = get_IDs(pa, K)
%GET_IDS get integer IDs corresponding to all combinations of limb types
% pa and K are the parent array and number of clusters per limb,
% respectively.
% nbh_IDs{i} lists the neighbours of part i
% target_IDs{i} gives an ID for the limb linking to each neighbour
% global_IDs{i}(j) gives a unique ID for part i when assigned type j; this
% is consistent with the scheme of assigning 0 to "background" in the
% training code.
p_no = length(pa);
nbh_IDs = cell(p_no, 1);
target_IDs = cell(p_no, 1);
for ii = 1:p_no
  for jj = 1:p_no
    is_child_of_jj = pa(ii) == jj;
    is_parent_of_jj = pa(jj) == ii;
    if is_child_of_jj || is_parent_of_jj
      nbh_IDs{ii} = cat(1, nbh_IDs{ii}, jj);
      if is_child_of_jj
          target_id = ii;
      else
          target_id = jj;
      end
      % For me, limbs are unidirectional, so I need to give a consistent ID
      % to each neighbouring limb. A good choice for the ID of a limb
      % therefore seems to be the ID of the child.
      target_IDs{ii} = cat(1, target_IDs{ii}, target_id);
    end
  end
end

global_IDs = cell(p_no, 1);
for p = 1:p_no
  global_IDs{p} = (p-1)*K+1:p*K;
end
