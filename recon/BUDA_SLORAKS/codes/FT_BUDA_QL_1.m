
function [A_ap, A_pa] =  FT_BUDA_QL_1(csm,wmap,esp,ky_idx_ap,ky_idx_pa)

%%
num_chan = size(csm,3);
nseg = length(esp);
nx = size(wmap,1);


esp=esp*1e-3; % in second 
Ry=6;

t_value_ap = esp(1)/Ry*ky_idx_ap';
t_value_pa = t_value_ap(end:-1:1);


N = size(wmap);
AccZ = 1;

%%
E = fftc(eye(N(2)),1);
E_ap = E(ky_idx_ap, : );
E_pa = E(ky_idx_pa, : );
PE_line = length(ky_idx_ap);

%% create and store encoding matrix
A_ap = zeross([num_chan*PE_line, N(2)*AccZ, N(1)]);
A_pa = zeross([num_chan*PE_line, N(2)*AccZ, N(1)]);

W_ap=zeross([PE_line,N(2),N(1),AccZ]);
W_pa=zeross([PE_line,N(2),N(1),AccZ]);
for ii_temp=1:AccZ
    W_ap(:,:,:,ii_temp)=exp(1i*mtimesx(repmat(t_value_ap,[1 1 N(1)]),permute(wmap(:,:,ii_temp),[3 2 1])));
    W_pa(:,:,:,ii_temp)=exp(1i*mtimesx(repmat(t_value_pa,[1 1 N(1)]),permute(wmap(:,:,ii_temp),[3 2 1])));
end
EW_ap = bsxfun(@times, repmat(E_ap,1,AccZ),reshape(permute(W_ap,[1,2,4,3]),[PE_line,N(2)*AccZ,N(1)]));
EW_pa = bsxfun(@times, repmat(E_pa,1,AccZ),reshape(permute(W_pa,[1,2,4,3]),[PE_line,N(2)*AccZ,N(1)]));

for c = 1:num_chan
    A_ap(1 + (c-1)*PE_line : c*PE_line, :, :) = bsxfun(@times,EW_ap , reshape(permute(csm(:,:,c,:),[3,2,4,1]),[1,N(2)*AccZ,N(1)]));
    A_pa(1 + (c-1)*PE_line : c*PE_line, :, :) = bsxfun(@times,EW_pa , reshape(permute(csm(:,:,c,:),[3,2,4,1]),[1,N(2)*AccZ,N(1)]));
end

%% operators A_ap & A_pa size 300 x 136 x 136  (Nc.Nseg x Ny x Nx)
%% kspace_ap & kspace_ap size 136 x 25 x 12  (Nc.Nseg x kx x ky)

end