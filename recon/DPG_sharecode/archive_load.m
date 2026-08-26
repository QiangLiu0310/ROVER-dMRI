function h = archive_load(archive, data_precision)

if nargin < 1 || isempty(archive)
  D = dir('ScanArchive*h5');
  for d = 1:numel(D)
    h = GERecon('Archive.Load', D(d).name);
    if strcmp(h.ScanType, 'Scan')
      h.h5file = D(d).name;
      continue;
    end
  end
else
  h = GERecon('Archive.Load', archive);
  h.h5file = archive;
end

if exist('data_precision','var') && ~isempty(data_precision)
  if strcmp(data_precision,'double') || strcmp(data_precision,'single')
    h.data_precision = data_precision;
  else
    error('pfile_load: data_precision must be ''single'' or ''double''');
  end
else
  h.data_precision = 'single';
end

if ~strcmp(h.ScanType, 'Autoshim')
  h.nShots    = h.DownloadData.rdb_hdr_rec.rdb_hdr_ileaves;
  h.nRawEchoes = h.DownloadData.rdb_hdr_rec.rdb_hdr_nechoes;
  h.nEchoes   = h.DownloadData.rdb_hdr_image.numecho;
  h.nKx       = h.DownloadData.rdb_hdr_rec.rdb_hdr_da_xres;
  h.nKy       = h.DownloadData.rdb_hdr_rec.rdb_hdr_da_yres-1;
  h.nover     = h.DownloadData.rdb_hdr_rec.rdb_hdr_hnover;
  h.nx        = h.DownloadData.rdb_hdr_image.dim_X;
  h.ny        = h.DownloadData.rdb_hdr_image.dim_Y;
  h.nRecX     = h.DownloadData.rdb_hdr_rec.rdb_hdr_rc_xres;
  h.nRecY     = h.DownloadData.rdb_hdr_rec.rdb_hdr_rc_yres;
  stoprcv     = h.DownloadData.rdb_hdr_rec.rdb_hdr_dab.stop_rcv;
  startrcv    = h.DownloadData.rdb_hdr_rec.rdb_hdr_dab.start_rcv;
  h.nChannels = stoprcv - startrcv + 1;
  
  h.nAcqsPerVol = recon_pass(h, 1, 'nacqspervol');
  h.nRawVols  =  recon_pass(h, 1, 'nvols');
  h.nVols     =  h.DownloadData.rdb_hdr_image.fphase;
  if h.nVols == 0
    h.nVols = h.nRawVols;
  end
  h.nPasses   =  recon_pass(h, 1, 'npasses');
  
  diffusionBit = bitget(h.DownloadData.rdb_hdr_rec.rdb_hdr_data_collect_type1, 22);
  
  h.isDiffusion = rcnhandle_getfield(h,'image','dfax') > 0;
  isIntegratedStaticPC = h.DownloadData.rdb_hdr_rec.rdb_hdr_ref == 5;
  isIntegratedRefBitSet = bitget(h.DownloadData.rdb_hdr_rec.rdb_hdr_pcctrl, 5);
  h.isIntegratedRef = isIntegratedStaticPC || isIntegratedRefBitSet || diffusionBit;
  
  for acqinvol = 1:h.nAcqsPerVol
    for z = 1:h.SlicesPerPass(acqinvol)
      sliceInfo = GERecon('Archive.Info', h, acqinvol, z);
      h.geometricSliceNumber(z,acqinvol) = sliceInfo.Number; % temporal slice location index
      h.sliceInfo(sliceInfo.Number) = sliceInfo; % spatial slice location index
    end
  end
  
end

% N.B.: while rhref = 5 should indicate integrated ref scan (in 1st vol), the scanner hangs pre DV26 if rhref = 5 and
% GE's online recon is off. ksepi.e uses ksepi_ghostcorr = 1 to enable blips-off ref scan as 1st extra volume, but rhref
% value will only be set to 5 if GE online recon and DV25 or earlier.
% This is why h.isIntegratedRef needs to be set to 1 if diffusionBit is 1, as bit 22 (rhtype1), which is used to flag
% for diffusion scan, follows ksepi_ghostcorr in the psd

end
