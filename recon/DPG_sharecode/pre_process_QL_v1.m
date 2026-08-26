function [image_pos, image_neg] = pre_process_QL_v4(irot, ishot)
% data pre-processing for mb2 data on the magnus scanner
% Qiang Liu qliu30@mgh.harvard.edu
% 02/27/2025
% 03/22/2025 debug version

%% Paths
base_path = ['/scratch/home/ql087/data_processing/rot_cimax/2026_06_30_invivo/3_shot/view', num2str(irot), '/'];

img_file_1  = sprintf('%srot_%d_dpg1.mat', base_path, irot);
img_file_2  = sprintf('%srot_%d_dpg2.mat', base_path,irot); 

%% Load the three shot EPI reference data
load(img_file_1);  
image_pos = reshape(ref, [400, 51, 28, 3, 2, 32]); clear ref
image_pos = squeeze(image_pos(:,:,:,ishot,[1 2],:));
% image_pos = image_pos(:,:,:,[1,3],:);

load(img_file_2);  
image_neg = reshape(ref, [400, 51, 28, 2, 32]); clear ref
image_neg = image_neg(:,:,:,1:2,:);
% image_neg = image_neg(:,:,:,[1,3],:);

end