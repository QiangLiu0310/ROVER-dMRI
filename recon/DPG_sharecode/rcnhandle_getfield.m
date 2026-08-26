
function hdrfield = rcnhandle_getfield(rcnhandle, group, field)

group = lower(group);

if ~exist('field','var')
  field = [];
end

if strcmp(rcnhandle.HandleType,'ScanArchive')
  group_rawhdr = 'rcnhandle.DownloadData.rdb_hdr_rec';
  group_rawhdrprefix = 'rdb_hdr_';
  group_exam = 'rcnhandle.DownloadData.rdb_hdr_exam';
  group_series = 'rcnhandle.DownloadData.rdb_hdr_series';
  group_image = 'rcnhandle.DownloadData.rdb_hdr_image';
else
  group_rawhdr = 'rcnhandle.header.RawHeader';
  group_rawhdrprefix = '';
  group_exam = 'rcnhandle.header.ExamData';
  group_series = 'rcnhandle.header.SeriesData';
  group_image = 'rcnhandle.header.ImageData';
  if ~isfield(rcnhandle,'header')
    error('rcnhandle_getfield: missing field ''header'' in rcnhandle');
  end
end

if strncmp(group, 'raw',3) || strcmp(group, 'rdb_hdr_rec')
  f = sprintf('hdrfield = %s.%s%s;', group_rawhdr,group_rawhdrprefix, field);
  fgroup = sprintf('hdrfield = %s;', group_rawhdr);
elseif strncmp(group, 'exam',4) || strcmp(group, 'rdb_hdr_exam')
  f = sprintf('hdrfield = %s.%s;', group_exam, field);
  fgroup = sprintf('hdrfield = %s;', group_exam);
elseif strncmp(group, 'series',4) || strcmp(group, 'rdb_hdr_series')
  f = sprintf('hdrfield = %s.%s;', group_series, field);
  fgroup = sprintf('hdrfield = %s;', group_series);
elseif strncmp(group, 'image',4) || strcmp(group, 'rdb_hdr_image')
  f = sprintf('hdrfield = %s.%s;', group_image, field);
  fgroup = sprintf('hdrfield = %s;', group_image);
end

try
  eval(f);
catch
  eval(fgroup);
end

if ischar(hdrfield)
  % cut at first null char
  hdrfield = strsplit(hdrfield, char(0));
  hdrfield = deblank(hdrfield{1});
elseif isstruct(hdrfield)
  fields = fieldnames(hdrfield);
  for j = 1:numel(fields)
    if ischar(hdrfield.(fields{j})) && isvector(hdrfield.(fields{j}))
      tmp = strsplit(hdrfield.(fields{j}), char(0));
      hdrfield.(fields{j}) = deblank(tmp{1});
    end
  end
end

end
