function y = interp( x, template, k ) 
% Whittaker-Shannon interpolation, approximately.  Instead of using
% sin(pi*x)/(pi*x), use sin(pi*x)/(T*tan(pi*x/T)).  Not the exact 
% interpolation formula (terms go to zero too quickly far away from the 
% center), but close enough for our purposes.  If k is 1 or 2, gives the
% first or second derivative of the interpolation wrt x

if size(x,1) == size(template,2) % In this case, we have precomputed the coordinates
    y = template*sinct(x,k)*(length(template)/2/pi)^k; 
else
    coord = interp_coord( x, size(template,2) );
    y = template*sinct(coord,k)*(length(template)/2/pi)^k;
end