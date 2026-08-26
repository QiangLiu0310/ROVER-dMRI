function res = ifft2c(x)

S = size(x);
fctr = S(1)*S(2);

x = reshape(x,S(1),S(2),prod(S(3:end)));

% res = zeros(size(x));
% for n=1:size(x,3)
res = sqrt(fctr)*fftshift(fftshift(ifft2(ifftshift(ifftshift(x,1),2)),1),2);
% end


res = reshape(res,S);

