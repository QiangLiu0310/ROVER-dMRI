function [ksp_new, kernel] = grappa2d(ksp,ksp_targ, calib, calib_targ, source_vec)
%%
% for regular grappa, calib_targ = [], which should be the same as calib; ksp_targ = [], which should be the same as ksp;
% for dpg, calib is single-polarity calib, and calib_targ is the dual-averaged calib
% Ndiag = 4; Nro = 9;  best results for Renzo's fat test 1
% Nkern = 3;  % three kernel needed
% Ndiag = 4;
% Nro = 9;    % kernel size in readout
% source_vec   : dims: Nkern, Nro, Ndiag(Nlin),Npar, Ndim

%% calib pos and neg

[Nkern,Nro,Nlin,Ndim] = size(source_vec);

if Ndim ~= 2
    error('dimension is not match; 2D axes required!')
end
if isempty(calib_targ)
    dpg_flag = 0;
    calib_targ = calib;
else
    dpg_flag = 1;
end

if isempty(ksp_targ)
    ksp_targ = ksp;
end

%% set up basic params and func
vec = @(x)x(:);
Nx_buff = [min(vec(source_vec(:,:,:,1))),max(vec(source_vec(:,:,:,1)))];
Ny_buff = [min(vec(source_vec(:,:,:,2))),max(vec(source_vec(:,:,:,2)))];

%% calib <- (nx ny nz nc)

[Nx_calib,Ny_calib,Nc] = size(calib);
[Y,X]=meshgrid(abs(Ny_buff(1))+1:Ny_calib-Ny_buff(2),abs(Nx_buff(1))+1:Nx_calib-Nx_buff(2));
target_ind = [X(:),Y(:)];
indt = sub2ind([Nx_calib,Ny_calib],vec(target_ind(:,1)),vec(target_ind(:,2)));

mask_calib = (calib_targ(:,:,:,1)~=0);
tmpl = mask_calib(indt)~=0;  % those point ~= 0
target_ind = target_ind(tmpl,:);
indt = indt(tmpl);

N = size(target_ind, 1);
source_ind = bsxfun(@plus,reshape(target_ind,1,N,1,1,Ndim),reshape(source_vec,Nkern,1,Nro,Nlin, Ndim));
vec = @(x)x(:);


inds = sub2ind([Nx_calib,Ny_calib],vec(source_ind(:,:,:,:,1)),vec(source_ind(:,:,:,:,2)));

calib_tmp = reshape(calib,[],Nc);
calib_targ_tmp = reshape(calib_targ,[],Nc);
source_data = calib_tmp(inds,:);
target_data = calib_targ_tmp(indt,:);

source_data = reshape(source_data,Nkern,[],Nro*Nlin, Nc);



for ii_k = 1:Nkern
    source_tmp = reshape(squeeze(source_data(ii_k,:,:,:,:,:)),[],Nro*Nlin*Nc);
    lambda = 0;
    tmp = pinv(source_tmp'*source_tmp+lambda*eye(Nro*Nlin*Nc));
    kernel{ii_k}=tmp*(source_tmp'*target_data);
end


%% apply kernel
[Nx,Ny,Nc] = size(ksp);
clear target_ind source_ind*

kspace_pad = padarray(ksp,[max(abs(Nx_buff)) max(abs(Ny_buff)) ]);
[Nx_pad,Ny_pad,~] = size(kspace_pad);
kspace_targ_pad = padarray(ksp_targ,[max(abs(Nx_buff)) max(abs(Ny_buff))]);
kspace_pad = reshape(kspace_pad,[],Nc);
kspace_targ_pad = reshape(kspace_targ_pad,[],Nc);

if dpg_flag == 0  % without dpg, the target is to fil in non-existing data
    mask = ksp_targ(:,:,1)==0;
else  % with dpg, this is to correct existing data
    mask = ksp_targ(:,:,1)~=0;
end
mask_pad = padarray(mask,[max(abs(Nx_buff)) max(abs(Ny_buff)) ]);
indt = find(vec(mask_pad)==1);  % find out the point that has no 

[target_ind(:,1),target_ind(:,2)] = ind2sub([Nx_pad,Ny_pad],indt);
N = size(target_ind, 1);
source_ind_tmp = bsxfun(@plus,reshape(target_ind,1,N,1,1,Ndim),reshape(source_vec,Nkern,1,Nro,Nlin,Ndim));

tmpl = [];
clear tmp
for ii_k = 1:Nkern
    inds = sub2ind([Nx_pad,Ny_pad],vec(source_ind_tmp(ii_k,:,:,:,1)),vec(source_ind_tmp(ii_k,:,:,:,2)));
    if dpg_flag == 0
        tmp = mask_pad(inds);
        tmp = reshape(tmp,N,[]);
        tmp = sum(tmp,2);
        tmp = find(tmp==0);
        source_ind = source_ind_tmp(ii_k,tmp,:,:,:,:);
        inds = sub2ind([Nx_pad,Ny_pad],vec(source_ind(1,:,:,:,1)),vec(source_ind(1,:,:,:,2)));
    else
        tmp = 1:N;
    end

    indt_tmp = indt(tmp,:);
    source_data = kspace_pad(inds,:);
    target_data = reshape(source_data,[],Nro*Nlin*Nc)*kernel{ii_k};
    kspace_targ_pad(indt_tmp,:) = target_data;
end

%%
kspace_targ_pad = reshape(kspace_targ_pad,Nx_pad,Ny_pad,Nc);
ksp_new = kspace_targ_pad(max(abs(Nx_buff))+1:Nx+max(abs(Nx_buff)),max(abs(Ny_buff))+1:Ny+max(abs(Ny_buff)),:);


