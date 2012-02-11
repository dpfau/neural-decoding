function coord = interp_coord( x, N ) 
% Given continuous time points and the period of a signal, returns matrix 
% of coordinates for reconstructing signal at those time points using
% Whittaker-Shannon interpolation

coord = ones(N,1)*x*N/(2*pi) - (1:N)'*ones(1,size(x,2));