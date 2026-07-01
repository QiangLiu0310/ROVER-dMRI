clear; close all; clc;

% For every nii/*.nii in the folder, keep only the 2nd volume (vol(:,:,:,2)),
% fix the header so it describes a single-frame 3D volume, and write the
% result to the nii_new folder.

baseDir = '/scratch/home/ql087/data_bwh/Philips/For_Qiang_phantom';
inDir   = fullfile(baseDir, 'nii');
outDir  = fullfile(baseDir, 'nii_b0');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

files = dir(fullfile(inDir, '*.nii'));

for k = 1:numel(files)
    fname   = files(k).name;
    inPath  = fullfile(inDir, fname);
    outPath = fullfile(outDir, fname);

    nii = MRIread(inPath);

    % keep only the 2nd volume
    nii.vol = nii.vol(:,:,:,1);

    % adjust the header to reflect a single-frame 3D volume
    nii.nframes = 1;
    nii.volsize = size(nii.vol);
    if isfield(nii, 'niftihdr') && isfield(nii.niftihdr, 'dim')
        nii.niftihdr.dim(1) = 3;   % number of dimensions
        nii.niftihdr.dim(5) = 1;   % size along the 4th (time) dimension
    end

    MRIwrite(nii, outPath);
    fprintf('Wrote %s\n', outPath);
end
