function parsave(fname, kspace_fill_full)
% PARSAVE  Save kspace_fill_full to fname. Use inside parfor (direct save not allowed).
    save(fname, 'kspace_fill_full', '-v7.3');
end
