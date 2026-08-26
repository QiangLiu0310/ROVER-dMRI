function res = fft2c(x)

S = size(x);
fctr = S(1)*S(2);

x = reshape(x,S(1),S(2),prod(S(3:end)));

% res = zeros(size(x));
res = 1/sqrt(fctr)*fftshift(fftshift(fft2(ifftshift(ifftshift(x,2),1)),1),2);

res = reshape(res,S);



