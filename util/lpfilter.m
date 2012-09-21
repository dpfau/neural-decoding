function y = lpfilter( x, sig )
% simple low-pass filter by convolving with a gaussian
filt = max(exp(-(0:size(x,1)-1).^2/sig^2), exp(-(-size(x,1):-1).^2/sig^2))';
z = fft(x);
y = ifft(z.*filt(:,ones(1,size(z,2))));