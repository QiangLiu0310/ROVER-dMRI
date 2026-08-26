function res = ifftc(x,dim)
 if nargin < 2
     dim = 1;
 end
 
S = size(x);

fctr = S(dim);
% for ii=1:size(x,3)
res = sqrt(fctr)*fftshift(ifft(ifftshift(x,dim),[],dim),dim);
% end

res = reshape(res, S);

end