function res = fftc(x, dim)
 if nargin < 2
     dim = 1;
 end
 
S = size(x);

fctr = S(dim);


% res = zeros(size(x));
res = 1/sqrt(fctr)*fftshift(fft(ifftshift(x,dim),[],dim),dim);

res = reshape(res,S);



