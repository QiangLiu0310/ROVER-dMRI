% ROVER pre-process for b0+b1000 data

addpath(genpath('/rfanfs/pnl-zorro/home/ql087/lq/Shared_MRI/npy-matlab-master'))

base_path='/scratch/home/ql087/data_bwh/Philips/For_Qiang_phantom/nii_new/';


output_path = fullfile(base_path, 'preprocess_rover_nii');
if ~exist(output_path, 'dir')
    mkdir(output_path);
end
step=1;

%% read the IJK2RAS matrix from the data

if (step==1)
    for i = 1:12

        f = dir(fullfile(base_path, sprintf('Dicom_DWI_rot%d_*.nii', i)));
        if isempty(f)
            warning('No nii found for rot%d, skipping.', i);
            continue;
        end
        nii = MRIread(fullfile(base_path, f(1).name));
        imgs_file = fullfile(output_path, sprintf('imgs_nii_%d.npy', i));
        writeNPY(nii.vol, imgs_file);
        rot = nii.vox2ras;
        affine_file = fullfile(output_path, sprintf('Affine_nii_%d.npy', i));
        writeNPY(rot, affine_file);

    end
end
