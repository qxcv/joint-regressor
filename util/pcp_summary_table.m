function summary_table = pcp_summary_table(pcps, limbs, limb_combos)
%PCP_SUMMARY_TABLE Make table summarising PCPs
% Will average over all limbs that need to be averaged over.
Limb = limb_combos.keys;
PCP = nan([length(Limb) 1]);
orig_names = {limbs.names};

for limb_idx=1:length(Limb)
    % Iterate over each combined limb
    combo_limb_name = Limb{limb_idx};
    
    % Grab the indices of the constituent limbs which make up this limb
    constit_limb_names = limb_combos(combo_limb_name);
    limb_mask = false([1 length(orig_names)]);
    for const_idx=1:length(constit_limb_names)
        limb_mask = limb_mask | strcmp(constit_limb_names(const_idx), orig_names);
    end
    
    % Now take the average of constituent PCPs
    relevant_pcps = pcps(limb_mask);
    PCP(limb_idx) = mean(relevant_pcps);
end

summary_table = table(PCP, 'RowNames', Limb');
end

