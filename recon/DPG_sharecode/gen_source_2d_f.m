function [source_vec]=gen_source_2d_f(Nro, Nlin, Ndim, AccY)
% Nlin = 2;
% Nro = 3;    % kernel size in readout
% Ndim = 2;   % this is for 2D case
kSize = [1,Nro];
R = AccY;
Nkern = R-1;  % number of 2D kernels
%
source_vec = zeros(Nkern, Nro,Nlin, Ndim);  % for dpg, currently, only do it in readout direction
diag_ro = -floor(Nro/2):floor(Nro/2);
% diag = reshape(-(Ndiag/2-1):(Ndiag/2),1,1,[]);
source_vec(:, :, :, 1) = repmat(diag_ro,[Nkern 1 Nlin]);

source_vec(:, :, 2, 2) = source_vec(:, :, 2, 2)+R;

for i = 1:R-1
    source_vec(i, :, :, 2) = source_vec(i, :, :, 2)-i;

    % % second kernel
    % source_vec(2, :, :, 2) = source_vec(2, :, :, 2)-2;
    %
    % % third kernel
    % source_vec(3, :, :, 2) = source_vec(3, :, :, 2)-3;

end

end
