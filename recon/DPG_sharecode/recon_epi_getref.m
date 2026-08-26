function [phaseCorrectionHandle, GE_refval] = recon_epi_getref(rcnhandle, refkSpace)

GE_refval = [];

ksz = size(refkSpace); ksz(end+1:6) = 1;

higher_order_flag = 0;
if higher_order_flag
    Ncoeff = 4;
else 
    Ncoeff = 2;
end

if nargin > 1
  
  echo = 1;
  
  if (ksz(2) < rcnhandle.nKy) % dynamic reflines
    % distribute reflines according to nShots
    newrefkSpace = zeros(ksz(1), ksz(2)*rcnhandle.nShots, ksz(3), ksz(4), ksz(5));
    for ky = 1:ksz(2)
      newrefkSpace(:,(ky-1)*rcnhandle.nShots+1,:,:,:) = refkSpace(:,ky,:,:,:);
    end
    refkSpace = newrefkSpace;
    ksz = size(refkSpace); ksz(end+1:6) = 1;
  end
  
  % Initialize an empty phase correction reference handle
  phaseCorrectionHandle = GERecon('Epi.LoadReference');
  % refkSpace may not include all slices
  emptySlices = squeeze(sum(sum(sum(sum(sum(abs(refkSpace),6),5),3),2),1))==0;
  GE_refval = zeros([1, ksz(2), ksz(3), sum(~emptySlices), Ncoeff], 'single');
  slice = 0;
  for z = 1:ksz(4)
    if ~emptySlices(z) % only compute refvals for non-empty slices
      slice = slice + 1;
      for channel = 1:ksz(3)
        % Compute phase correction coefficients, store in reference handle
        if higher_order_flag
          GE_refval(1,:,channel, slice, :) = GERecon('Epi.ComputeCoefficients', refkSpace(:,:,channel,z,echo), 1);
        else
          GE_refval(1,:,channel, slice, :) = GERecon('Epi.ComputeCoefficients', phaseCorrectionHandle, refkSpace(:,:,channel,z,echo), z, channel);
        end
      end
    end
  end
    
else % no refkSpace given
  try
    
    if strcmp(rcnhandle.HandleType, 'ScanArchive')
      % if ref.h5 does not exist, an error will be thrown and we'll end up in the catch block
      reffile = '/usr/g/bin/ref.h5';
      h5info(rcnhandle.h5file, reffile);
    else
      reffile = '/usr/g/bin/ref.dat';
      if ~exist(reffile,'file')
        reffile = './ref.dat';
      end
      if ~exist(reffile,'file')
        error('recon_epi_getref: could not find %s', reffile);
      end
    end
    
    % Read phase correction coefficients from ref.h5 or ref.dat
    phaseCorrectionHandle = GERecon('Epi.LoadReference', reffile);
    
  catch
    phaseCorrectionHandle = [];
  end
  
end

end

