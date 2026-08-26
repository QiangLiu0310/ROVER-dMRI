clc; clear; close all

for irot=5:6
    save_dir=['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/3_shot/view', num2str(irot), '/'];
    load(['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/gre/csm_v' num2str(irot), '.mat'])
    % receive = flip(receive,1);
    % receive = flip(receive,2); % for rot 1, only this 
    sens_gre_gcc = receive; clear receive
    esp=ones(51,1);
    tmp=[1:2:28 2:2:28];[~,r]=sort(tmp);
    load(['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/3_shot/view', num2str(irot), '/dpg_cor_shot_1_dwi_2.mat']) % shot1
    kspace_cor_ap1=kspace_fill_full(:,95:6:end,:,r);
    load(['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/3_shot/view', num2str(irot), '/dpg_cor_shot_2_dwi_2.mat']) % shot1
    kspace_cor_ap2=kspace_fill_full(:,95:6:end,:,r);
    load(['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/3_shot/view', num2str(irot), '/dpg_cor_shot_3_dwi_2.mat']) % shot1
    kspace_cor_ap3=kspace_fill_full(:,95:6:end,:,r);
    z=[2:2:28 1:2:28]; [~,r1]=sort(z);
    kspace_cor_ap1=kspace_cor_ap1(:,:,:,r1);
    kspace_cor_ap2=kspace_cor_ap2(:,:,:,r1);
    kspace_cor_ap3=kspace_cor_ap3(:,:,:,r1);
    clear z r1 r kspace_fill_full
    ky_idx_ap_shot1=95:6:400;
    ky_idx_ap_shot2=97:6:400;
    ky_idx_ap_shot3=99:6:400;
    ky_idx_shots = {ky_idx_ap_shot1, ky_idx_ap_shot2, ky_idx_ap_shot3};

    Niter = 20;   % iterations of CG
    tol = 1e-4;
    Display = 1;

    use_coil_compression = true;
    num_cc = 20;

    sc = 0.5*(max(abs(kspace_cor_ap1(:))) + max(abs(kspace_cor_ap2(:))) + max(abs(kspace_cor_ap3(:))));
    kspace_cor_ap1 = kspace_cor_ap1./sc;
    kspace_cor_ap2 = kspace_cor_ap2./sc;
    kspace_cor_ap3 = kspace_cor_ap3./sc;
    
    % Reconstruction dimensions
    nx = 400;
    ny = 400;
    kx = 400;
    ky = 51;
    num_shot = 3;
    num_slice = size(kspace_cor_ap1,4);

    SENSE_all = zeros(nx,ny,num_shot,num_slice,'like',kspace_cor_ap1);
    S_LORAKS_all = zeros(nx,ny,num_shot,num_slice,'like',kspace_cor_ap1);

    tic

    for slice =  1:num_slice
        kdata = cat(4, kspace_cor_ap1(:,:,:,slice), kspace_cor_ap2(:,:,:,slice), kspace_cor_ap3(:,:,:,slice));

        % Coil sensitivity maps for this slice
        csm = squeeze(sens_gre_gcc(:,:,slice,:));

        % Optional coil compression (apply same compression to csm and k-space)
        if use_coil_compression
            [csm_k_cc, cmp_mtx] = coil_compress2d_local(csm, num_cc);
            num_chan_cc = size(cmp_mtx,2);
            kdata_cc = zeros(size(kdata,1), size(kdata,2), num_chan_cc, size(kdata,4), 'like', kdata);
            for shot_idx = 1:size(kdata,4)
                kdata_cc(:,:,:,shot_idx) = coil_apply2d_local(kdata(:,:,:,shot_idx), cmp_mtx);
            end
            kdata = kdata_cc;
            csm=csm_k_cc;
        end

        num_chan = size(kdata,3);

        % Circular mask
        tmp = squeeze(sum(abs(kdata),3));
        mask = zeros(size(tmp),'like',tmp);
        mask(tmp>0) = 1;
        mask = repmat(mask,[1 1 1 num_chan]);
        mask = permute(mask,[1 2 4 3]);

        % 3-shot same-polarity encoding (no AP/PA BUDA pair model).
        At = @(x) transpose_ms_sense_ql(x,csm,ky_idx_shots,mask,nx,ny,num_chan,num_shot);
        A = @(x) forward_ms_sense_ql(x,csm,ky_idx_shots,mask,nx,ny,num_chan,num_shot);
        AtA = @(x) At(A(x));
        Ahd = At(kdata);

        SENSE_init = CG_Recon(AtA,Ahd,Niter,tol,Display);

        %% S-LORAKS
        ns = num_shot;
        LORAKS_type = 1;

        z = SENSE_init;
        R = 3;
        rank = 80; % 80
        lambda = 1e-3; % 1e-3
        tol_loraks = 1e-4;
        max_iter = 8;

        [in1,in2] = meshgrid(-R:R,-R:R);
        idx = find(in1.^2 + in2.^2 <= R^2);
        patchSize = numel(idx);

        coil_sens = repmat(csm,[1 1 1 num_shot]);
        coil_sens = permute(coil_sens,[1 2 4 3]);

        B = @(x) ft2(coil_sens.*repmat(x,[1 1 1 num_chan]));
        Bh = @(x) sum(conj(coil_sens).*ift2(x),4);

        P_M = @(x) LORAKS_operators(x,nx,ny,ns*num_chan,R,LORAKS_type,[]);

        ns = num_shot;
        N1 = nx;
        N2 = ny;
        Nc = num_chan*ns;

        ZD = @(x) padarray(reshape(x,[N1 N2 Nc]),[2*R, 2*R], 'post');
        ZD_H = @(x) x(1:N1,1:N2,:,:);

        for iter = 1:max_iter
            z_cur = z;
            pz = B(z);
            MM = P_M(pz);
            l = sum(MM,1);
            l = find(abs(l)>0);
            MM = MM(:,l);

            Um = svd_left(MM);
            nmm = Um(:,rank+1:end)';
            Bhr = 0;
            LhL = @(x) zeros(size(x),'like',x);

            if LORAKS_type == 1
                nf = size(nmm,1);
                nmm = reshape(nmm,[nf, patchSize, 2*Nc]);
                nss_h = reshape(nmm(:,:,1:2:end)+1j*nmm(:,:,2:2:end),[nf, patchSize*Nc]);
                Nis = filtfilt(nss_h,'C',N1,N2,Nc,R);
                Nis2 = filtfilt(nss_h,'S',N1,N2,Nc,R);

                L1 = @(x) ZD_H(ifft2(squeeze(sum(Nis.*repmat(fft2(ZD(B(x))),[1 1 1 Nc]),3))));
                L2 = @(x) ZD_H(ifft2(squeeze(sum(Nis2.*repmat(conj(fft2(ZD(B(x)))),[1 1 1 Nc]),3))));
                LhL = @(x) 2*Bh(reshape(L1(x)-L2(x),[nx ny ns num_chan]));
            end

            M = @(x) AtA(x) + lambda*LhL(x);
            z = CG_Recon(M,Ahd+lambda*Bhr,Niter,tol_loraks,Display);

            t = norm(z_cur(:)-z(:))/max(norm(z(:)),eps);
            if t < tol_loraks
                break;
            end
        end

        SENSE_all(:,:,:,slice) = SENSE_init;
        S_LORAKS_all(:,:,:,slice) = z;

        disp(['finish slice: ', num2str(slice), '/', num2str(num_slice)])
    end
    toc

    SENSE_recon = SENSE_all;
    S_LORAKS_recon = S_LORAKS_all;

    slice_show = min(14,28);

    kk_sense = fft2c(SENSE_recon(:,:,:,slice_show));
    figure,imagesc(log(abs(kk_sense(:,:)))), colormap gray
    title('SENSE kspace (3-shot)')
    figure,imagesc(cat(2,rot90(abs(SENSE_recon(:,:,1,slice_show)),3),rot90(abs(SENSE_recon(:,:,2,slice_show)),3),rot90(abs(SENSE_recon(:,:,3,slice_show)),3))), colormap gray
    title('SENSE recon (shot1/shot2/shot3)')

    kk_sloraks = fft2c(S_LORAKS_recon(:,:,:,14));
    figure,imagesc(log(abs(kk_sloraks(:,:)))), colormap gray
    title('S-LORAKS kspace (3-shot)')
    figure,imagesc(cat(2,rot90(abs(S_LORAKS_recon(:,:,1,slice_show)),3),rot90(abs(S_LORAKS_recon(:,:,2,slice_show)),3),rot90(abs(S_LORAKS_recon(:,:,3,slice_show)),3))), colormap gray
    title('S-LORAKS recon (shot1/shot2/shot3)')
    out_mat = fullfile(save_dir, 'SENSE_S_LORAKS_b0.mat');

    save(out_mat, 'SENSE_recon', 'S_LORAKS_recon', '-v7.3');
end

function [kdata_cc, cmp_mtx] = coil_compress2d_local(kdata_in, num_out)
[nx, ny, nc] = size(kdata_in);
num_out = min(num_out, nc);
X = reshape(kdata_in, [], nc);
[~, ~, V] = svd(X, 'econ');
cmp_mtx = V(:,1:num_out);
Xc = X * cmp_mtx;
kdata_cc = reshape(Xc, nx, ny, num_out);
end

function kdata_cc = coil_apply2d_local(kdata_in, cmp_mtx)
[nx, ny, nc] = size(kdata_in);
X = reshape(kdata_in, [], nc);
Xc = X * cmp_mtx;
kdata_cc = reshape(Xc, nx, ny, size(cmp_mtx,2));
end

function kdata = forward_ms_sense_ql(Im,csm,ky_idx_shots,mask,nx,ny,num_chan,num_shot)
kdata = zeros(nx, numel(ky_idx_shots{1}), num_chan, num_shot, 'like', Im);
for s = 1:num_shot
    ky_idx = ky_idx_shots{s};
    for c = 1:num_chan
        im_c = csm(:,:,c) .* Im(:,:,s);
        k_full = ft2(im_c);
        k_shot = k_full(:,ky_idx);
        kdata(:,:,c,s) = k_shot .* mask(:,:,c,s);
    end
end
end

function Im = transpose_ms_sense_ql(kdata,csm,ky_idx_shots,mask,nx,ny,num_chan,num_shot)
Im = zeros(nx, ny, num_shot, 'like', kdata);
for s = 1:num_shot
    ky_idx = ky_idx_shots{s};
    for c = 1:num_chan
        k_full = zeros(nx, ny, 'like', kdata);
        k_full(:,ky_idx) = kdata(:,:,c,s) .* mask(:,:,c,s);
        im_c = ift2(k_full);
        Im(:,:,s) = Im(:,:,s) + conj(csm(:,:,c)) .* im_c;
    end
end
end
