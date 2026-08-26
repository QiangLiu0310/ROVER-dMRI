close all; clear all ; clc

%%
save_path='/scratch/home/ql087/data_processing/rot_cimax/2026_06_17_phantom/';
data_path = '/scratch/home/ql087/data_bwh/Cima_data/2026_06_18_mgh_phantom/'
filename1 = 'meas_MID00062_FID28329_pulseq_te74_r2_3shot_rot4'; 

for iRot=1:12

    D=dir([data_path filename1]);
    [~,I]=sort([D(:).datenum]);
    twix_obj = mapVBVD([data_path filename1]);

    %% dpg 1+2

    if iscell(twix_obj)
        rawdata0 = double(twix_obj{end}.image.unsorted());
    else
        rawdata0 = double(twix_obj.image.unsorted());
    end
    rawdata0=single(rawdata0);

    rawdata0=reshape(rawdata0,[468 32 51 28 29 3]); % fe coil pe sli diff+dpg rot
    rawdata=rawdata0(:,:,:,:,1:27,iRot);
    rawdata=reshape(rawdata,[468 32 51*28*27]);
    rawdata=single(rawdata);
    rawdata1=rawdata0(:,:,:,:,28:29,iRot);
    rawdata1=reshape(rawdata1,[468 32 51*28*2]);
    rawdata1=single(rawdata1);
    clear rawdata0

    seq = mr.Sequence();
    read(seq,'/rfanfs/pnl-zorro/home/ql087/lq/Tests/Test_16_Jun_17/recon/ep2d_6mm_TE74_R2_3shots_slc28_dir6_rot1_recon_1.seq')
    traj_recon_delay=0e-6;
    [ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP('trajectory_delay', traj_recon_delay);
    %% automatic detection of the measurement parameters (FOV, matrix size, etc)
    nADC = size(rawdata, 1);
    k_last=ktraj_adc(:,end);
    k_2last=ktraj_adc(:,end-nADC);
    delta_ky=k_last(2)-k_2last(2);
    fov=1/abs(delta_ky);

    %% manually (re-)define FOV and resolution
    Nx=400;
    Ny=400;
    Ny_sampled=51;
    slice_num=28;

    %% analyze the trajecotory, resample the data
    nCoils = size(rawdata, 2); % the incoming data order is [kx coils acquisitions]
    nAcq=size(rawdata,3);
    nD=size(ktraj_adc, 1);

    kxmin=min(ktraj_adc(1,:));
    kxmax=max(ktraj_adc(1,:));
    kxmax1=kxmax/(Nx/2-1)*(Nx/2); % this compensates for the non-symmetric center definition in FFT
    kmaxabs=max(kxmax1, -kxmin);

    kxx= ((-Nx/2):(Nx/2-1))/(Nx/2)*kmaxabs; % kx-sample positions

    ktraj_adc2=reshape(ktraj_adc,[size(ktraj_adc,1), nADC, size(ktraj_adc,2)/nADC]);
    t_adc2=reshape(t_adc,[nADC, length(t_adc)/nADC]);

    data_resampled=zeros(length(kxx),nCoils, nAcq);

    tic
    parfor t_c=1:size(data_resampled,3)
        % for a=1:nAd
            for c=1:nCoils
                data_resampled(:,c,t_c)=interp1(ktraj_adc2(1,:,t_c),rawdata(:,c, t_c),kxx,'spline',0);
            end
        % end
    end
    toc

    % delete(gcp('nocreate'))

    disp('interporlation completed for std data')

    % data_resampled=reshape(data_resampled,[length(kxx), nCoils, nAcq]);

    diff_plus_dum=27;
    %% save the reference data
    ref=permute(data_resampled,[1 3 2]);
    ref=reshape(ref,[Nx Ny_sampled slice_num*diff_plus_dum nCoils]);
    ref=single(ref);
    save([save_path 'rot_', num2str(iRot+0), '_dpg1.mat'], 'ref', '-v7.3')

    clear ref data_resampled ref

    %% dpg 2
    seq = mr.Sequence();
    read(seq,'/rfanfs/pnl-zorro/home/ql087/lq/Tests/Test_16_Jun_17/recon/ep2d_6mm_TE74_R2_3shots_slc28_dir6_rot1_recon_2.seq')
    traj_recon_delay=0e-6;
    [ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP('trajectory_delay', traj_recon_delay);
    %% automatic detection of the measurement parameters (FOV, matrix size, etc)
    nADC = size(rawdata1, 1);
    k_last=ktraj_adc(:,end);
    k_2last=ktraj_adc(:,end-nADC);
    delta_ky=k_last(2)-k_2last(2);
    fov=1/abs(delta_ky);

    nAcq=size(rawdata1,3);
    nD=size(ktraj_adc, 1);

    kxmin=min(ktraj_adc(1,:));
    kxmax=max(ktraj_adc(1,:));
    kxmax1=kxmax/(Nx/2-1)*(Nx/2); % this compensates for the non-symmetric center definition in FFT
    kmaxabs=max(kxmax1, -kxmin);

    kxx= ((-Nx/2):(Nx/2-1))/(Nx/2)*kmaxabs; % kx-sample positions

    ktraj_adc2=reshape(ktraj_adc,[size(ktraj_adc,1), nADC, size(ktraj_adc,2)/nADC]);
    t_adc2=reshape(t_adc,[nADC, length(t_adc)/nADC]);

    diff_plus_dum=2;
    nAd=nAcq/diff_plus_dum;
    data_resampled=zeros(length(kxx),nCoils, nAd, diff_plus_dum);
    rawdata1=reshape(rawdata1,[size(rawdata1,1)  size(rawdata1,2) nAd diff_plus_dum]);


    tic
    parfor t_c=1:size(data_resampled,4)
        for a=1:nAd
            for c=1:nCoils
                data_resampled(:,c,a,t_c)=interp1(ktraj_adc2(1,:,a),rawdata1(:,c,a, t_c),kxx,'spline',0);
            end
        end
    end
    toc

    % delete(gcp('nocreate'))

    disp('interporlation completed for std data')

    data_resampled=reshape(data_resampled,[length(kxx), nCoils, nAcq]);


    %% save the reference data
    ref=permute(data_resampled,[1 3 2]);
    ref=reshape(ref,[Nx Ny_sampled slice_num*diff_plus_dum nCoils]);
    ref=single(ref);

    save([save_path 'rot_', num2str(iRot+0), '_dpg2.mat'], 'ref', '-v7.3')


    clearvars -except iRot save_path data_path filename1
end

