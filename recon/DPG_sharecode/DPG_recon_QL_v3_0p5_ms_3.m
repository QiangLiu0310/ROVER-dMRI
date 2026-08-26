% DPG for single-band in plane 3-shot
% Jun 18 2026

clear;close all;clc;
addpath('/rfanfs/pnl-zorro/home/ql087/lq/Shared_MRI/DPG_sharecode/orchestra.matlab')

for irot=[1:12]
    fprintf('Processing rot %d\n', irot);

    for ishot = 1:3
        fprintf('Processing shot %d\n', ishot);
        save_path=['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/3_shot/view', num2str(irot),'/'];
        %% use GPU
        useGPU=true;
        if useGPU
            try
                choose_GPU;
            catch
                gpudev = gpuDevice(1); 
            end
        end
        %% run data preprocessing
        [ref_pos, ref_neg] = pre_process_QL_v1(irot,ishot);
        %% data setup
        Nx = 400;
        Ny = 400;
        Ny_sampled = 51;
        Nz = 28;
        Nt = 51 ;
        Ncoils = 32;
        shot = 1;
        st = Ny-6*Nt;
        AccY = 6;
        AccZ = 1;
        Nz_acq = Nz/AccZ;
        slc_reorder_calib = [1:2:Nz,2:2:Nz];

        %% reorder

        ref_pos=reshape(ref_pos,[Nx, Ny_sampled, Nz, shot, 2, Ncoils]);
        ref_pos = ref_pos(:,:,slc_reorder_calib,:,:,:);
        ref_neg=reshape(ref_neg,[Nx, Ny_sampled, Nz, shot, 2, Ncoils]);
        ref_neg = ref_neg(:,:,slc_reorder_calib,:,:,:);
        load(['/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/3_shot/view', num2str(irot), '/rot_', num2str(irot), '_dpg1.mat']);
        ref=reshape(ref,[Nx, Ny_sampled, Nz, 3, 9, Ncoils]);

        % dwi_list = [2 4:size(ref,5)];
        dwi_list = [2 4]; 
        image = ref(:,:,slc_reorder_calib,ishot,dwi_list,:);

        ref_pos_caipicorr = ref_pos;
        ref_neg_caipicorr = ref_neg;
        clear ref_pos ref_neg

        %% do ghost estimation
        disp('ghost estimation')
        current_path = pwd;
        shell_path = '/rfanfs/pnl-zorro/home/ql087/lq/Pulseq_V151_ROVER_dMRI/recon/DPG_sharecode/ksepi_1mm';

        cd(shell_path);
        prot1 = archive_load();

        item_select_ref_delay = 1;

        for i = 1:shot
            [phaseCorrectionHandle, GE_refval] = recon_epi_getref(prot1, permute(ref_pos_caipicorr(:,:,:,i,item_select_ref_delay,:),[1 2 6 3 4 5]));
            theta0_pos(:,:,:,:,i) = permute(GE_refval(:,:,:,:,1),[1 2 4 3]);
            theta1_pos(:,:,:,:,i) = permute(GE_refval(:,:,:,:,2),[1 2 4 3]);
        end

        for i = 1:shot
            [phaseCorrectionHandle, GE_refval] = recon_epi_getref(prot1, permute(ref_neg_caipicorr(:,:,:,i,item_select_ref_delay,:),[1 2 6 3 4 5]));
            theta0_neg(:,:,:,:,i) = permute(GE_refval(:,:,:,:,1),[1 2 4 3]);
            theta1_neg(:,:,:,:,i) = permute(GE_refval(:,:,:,:,2),[1 2 4 3]);
        end

        cd(current_path)

        %% correct slice order, make the order linear in SI
        disp('create ref data')
        item_select_ref = 2;
        ref_data_pos = zeros(Nx,Ny,Nz,Ncoils);
        for i = 1:shot
            ref_data_pos(:,(st+i):AccY:end,:,:) = ref_pos_caipicorr(:,:,:,i,item_select_ref,:);
        end

        ref_data_neg = zeros(Nx,Ny,Nz,Ncoils);
        for i = 1:shot
            ref_data_neg(:,(st+i):AccY:end,:,:) = ref_neg_caipicorr(:,:,:,i,item_select_ref,:);
        end

        %% phase correction estimation
        xind = -Nx/2:Nx/2-1;
        ref_pos_corr = zeros(Nx,Ny,Nz,Ncoils);
        ref_pos_semicorr = zeros(Nx,Ny,Nz,Ncoils);

        x_shift=0;
        phs_extra_ro = exp(1i*xind.'*x_shift);


        for i = 1:shot
            ref_pos_corr(:,(st+i):AccY:end,:,:) = fftc(ifftc(ref_data_pos(:,(st+i):AccY:end,:,:),1) .*exp(1i*(theta0_pos(:,:,:,:,i)+xind(:).*theta1_pos(:,:,:,:,i))).*phs_extra_ro,1);
            ref_pos_semicorr(:,(st+i):AccY:end,:,:) = fftc(ifftc(ref_data_pos(:,(st+i):AccY:end,:,:),1) .*exp(1i*(theta0_pos(:,:,:,:,i)+xind(:).*theta1_pos(:,:,:,:,i))),1);
        end

        ref_neg_corr = zeros(Nx,Ny,Nz,Ncoils);
        ref_neg_semicorr = zeros(Nx,Ny,Nz,Ncoils);

        for i = 1:shot
            ref_neg_corr(:,(st+i):AccY:end,:,:) = fftc(ifftc(ref_data_neg(:,(st+i):AccY:end,:,:),1) .*exp(1i*(theta0_neg(:,:,:,:,i)+xind(:).*theta1_neg(:,:,:,:,i))).*conj(phs_extra_ro),1);
            ref_neg_semicorr(:,(st+i):AccY:end,:,:) = fftc(ifftc(ref_data_neg(:,(st+i):AccY:end,:,:),1) .*exp(1i*(theta0_neg(:,:,:,:,i)+xind(:).*theta1_neg(:,:,:,:,i))),1);
        end

        %% DPG step 1, synthesize sms target data from reference data

        ref_pos_corr_sms = zeros(Nx,Ny,Nz_acq,Ncoils);
        ref_neg_corr_sms = zeros(Nx,Ny,Nz_acq,Ncoils);
        ref_pos_semicorr_sms = zeros(Nx,Ny,Nz_acq,Ncoils);
        ref_neg_semicorr_sms = zeros(Nx,Ny,Nz_acq,Ncoils);

        for i = 1
            ref_pos_corr_sms(:,(st+i):AccY:end,:,:) = ref_pos_corr(:,(st+i):AccY:end,1:Nz_acq,:);
            ref_neg_corr_sms(:,(st+i):AccY:end,:,:) = ref_neg_corr(:,(st+i):AccY:end,1:Nz_acq,:);
            ref_pos_semicorr_sms(:,(st+i):AccY:end,:,:) = ref_pos_semicorr(:,(st+i):AccY:end,1:Nz_acq,:);
            ref_neg_semicorr_sms(:,(st+i):AccY:end,:,:) =ref_neg_semicorr(:,(st+i):AccY:end,1:Nz_acq,:);
        end

        %% actual mb data correction (all selected DWI in this shot)
        mb2_k_pos_caipicorr = image;
        clear image
        xind = -Nx/2:Nx/2-1;
        phs_extra_ro = 1;
        phs_extra_pe = ones(1,Ny_sampled);
        %% sms param
        pes_index = 0;
        slc_unalias=1:Nz_acq;
        kspace_calib = (ref_pos_corr + ref_neg_corr )/2;

        kcalib = permute(kspace_calib,[1 2 4 3]);  % this is the training data for parallel imaging
        kcalib_avg = permute(kspace_calib,[1 2 4 3]);  % this is the training data for parallel imaging

        %% source_vec for SMS and dpg
        disp('GRAPPA prep')
        recon_flag = 'GRAPPA'

        Nkern = 1;
        Nlin = 1;
        Npar = 1;
        Nro = 5;
        Ndim = 2;
        kSize = [1,Nro];
        source_vec_dpg = zeros(Nkern, Nro,Nlin, Ndim);
        diag_ro = -floor(Nro/2):floor(Nro/2);
        source_vec_dpg(1,:,1,1,1) = diag_ro;

        %% DPG step 2: generate paired source and target data

        kcalib_pos = permute(ref_pos_semicorr_sms,[1 2 4 3]);
        kcalib_neg = permute(ref_neg_semicorr_sms,[1 2 4 3]);

        kcalib_pos_sms = permute(ref_pos_corr_sms,[1 2 4 3]);
        kcalib_neg_sms = permute(ref_neg_corr_sms,[1 2 4 3]);


        kcalib_dpg = cat(5,kcalib_pos_sms,kcalib_neg_sms);
        kcalib_dpg_avg = mean(kcalib_dpg,5);

        pos_vec = st+(1:2*AccY:Nt*AccY);
        neg_vec = st+(AccY+1:2*AccY:Nt*AccY);

        ksrc_pos = kcalib_pos(:,pos_vec(:),:,:);
        ktgt_pos = kcalib_dpg_avg(:,pos_vec(:),:,:);

        ksrc_neg = kcalib_pos(:,neg_vec(:),:,:);
        ktgt_neg = kcalib_dpg_avg(:,neg_vec(:),:,:);

        if(ishot==1)
            save([save_path, 'shot1_acs','.mat'],'ksrc_neg', 'ksrc_pos',  'ktgt_pos',  'ktgt_neg', '-v7.3')
        else
            clear ksrc_neg ksrc_pos ktgt_pos ktgt_neg
            load([save_path, 'shot1_acs.mat'])
        end


        for idwi_idx = 1:numel(dwi_list)
            idwi = dwi_list(idwi_idx);
            mb2_pos_corr = zeros(Nx,Ny,Nz_acq,Ncoils);
            mb2_k_pos_curr = squeeze(mb2_k_pos_caipicorr(:,:,:,1,idwi_idx,:));
            for i = 1:shot
                mb2_pos_corr(:,(st+i):AccY:end,:,:) = fftc(ifftc(mb2_k_pos_curr,1).*exp(1i*(theta0_pos(:,:,[1:Nz_acq],:,i)+xind(:).*theta1_pos(:,:,[1:Nz_acq],:,i))).*phs_extra_ro.*phs_extra_pe,1);
            end
            kspace_cor = permute(mb2_pos_corr,[1 2 4 3]);

            %% DPG correction
            kspace_fill_full=zeros(Nx,Ny,Ncoils,Nz_acq);
            for ii_slc = 1:ceil(Nz/AccZ)

                slc = slc_unalias(:,ii_slc,1);
                kspace_fill = zeros(Nx,Ny,Ncoils);

                ksrc_pos_tmp =ksrc_pos(:,:,:,ii_slc);
                ksrc_neg_tmp =ksrc_neg(:,:,:,ii_slc);

                ktgt_pos_tmp =ktgt_pos(:,:,:,ii_slc);
                ktgt_neg_tmp =ktgt_neg(:,:,:,ii_slc);

                pos_vec_data = st+(1:AccY*2:Nt*AccY);
                neg_vec_data = st+((AccY+1):AccY*2:Nt*AccY);
                ksp_pos = zeros(Nx,Ny,Ncoils);
                ksp_neg = zeros(Nx,Ny,Ncoils);

                for ii_segnum = 5 % test from 1 -5 pairs of kernels
                    Nseg = ii_segnum*2-1; % number of readout segment

                    for ii_seg = 1:Nseg
                        l_seg = ceil(Nx/Nseg)*(ii_seg-1)+1:ceil(Nx/Nseg)*ii_seg;
                        l_seg(l_seg>Nx) = [];

                        kspace_targ = [];

                        [ksp_pos(l_seg,pos_vec_data(:),:), kernel_pos] = grappa2d(kspace_cor(l_seg,pos_vec_data(:),:,ii_slc), kspace_targ, ksrc_pos_tmp(l_seg,:,:),ktgt_pos_tmp(l_seg,:,:),source_vec_dpg);
                        [ksp_neg(l_seg,neg_vec_data(:),:), kernel_neg] = grappa2d(kspace_cor(l_seg,neg_vec_data(:),:,ii_slc), kspace_targ, ksrc_neg_tmp(l_seg,:,:),ktgt_neg_tmp(l_seg,:,:),source_vec_dpg);

                    end
                    kspace_fill(:,pos_vec_data(:),:) = ksp_pos(:,pos_vec_data(:),:);
                    kspace_fill(:,neg_vec_data(:),:) = ksp_neg(:,neg_vec_data(:),:);

                end

                kspace_fill_full(:,:,:,ii_slc)=kspace_fill;
            end
            kspace_fill_full=single(kspace_fill_full);
            save([save_path,'dpg_cor_shot_', num2str(ishot), '_dwi_', num2str(idwi), '.mat'], 'kspace_fill_full', '-v7.3')
        end
    end
end