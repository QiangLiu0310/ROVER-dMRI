function result = recon_pass(archive, oneBasedPassIndex, whatToGet)

if oneBasedPassIndex < 1
  error('archive_pass: Pass index (2nd arg) must be >= 1');
end

if strcmp(archive.HandleType, 'ScanArchive')
  nVols = archive.DownloadData.rdb_hdr_rec.rdb_hdr_reps;
  nPasses = archive.DownloadData.rdb_hdr_rec.rdb_hdr_npasses;  
else
  nVols = archive.header.RawHeader.reps;
  nPasses = archive.header.RawHeader.npasses;
end
nAcqsPerVolume = nPasses / nVols;

acqIndexInVolume = mod(oneBasedPassIndex-1, nAcqsPerVolume) + 1;
volumeIndex      = floor((oneBasedPassIndex-1) / nAcqsPerVolume) + 1;

if strncmpi(whatToGet, 'npasses', 7) % Number of passes total
  result = nPasses;
elseif strncmpi(whatToGet, 'nacqspervol', 11) % Number of acqs per volume
  result = nAcqsPerVolume;
elseif strncmpi(whatToGet, 'acqindxvol', 10) % Which acqindx in volume [0,acqspervol)
  result = acqIndexInVolume;
elseif strncmpi(whatToGet, 'volindx', 4) % Which volume index
  if volumeIndex > nVols
    error('recon_pass: Pass index (2nd arg) exceeds scan');
  end
  result = volumeIndex;
elseif strncmpi(whatToGet, 'nvols', 4) % How many volumes
  result = nVols;
else
  error('recon_pass: 3rd arg must be one of: nacqspervol, acqindxvol, volindx, nvols');
end

end
