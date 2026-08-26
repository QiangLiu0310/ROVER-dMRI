function parsave_recon(fname, I_fill, I_pocs)
% PARSAVE_RECON  Save I_fill and I_pocs to fname. Use inside parfor (direct save not allowed).
    save(fname, 'I_fill', 'I_pocs', '-v7.3');
end
