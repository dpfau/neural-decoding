function ll = data_log_lik( data, map, prec, params )
% Computes the marginal likelihood of the data by use of the Laplace
% approximation to the posterior of the latent path given the data.  From
% basic probability:
% log( p( data ) ) = log( p( data, path ) ) - log( p ( path | data ) )
% The joint probability p( data, path ) can be calculated analytically.
% The posterior probability cannot be calculated analytically, but the
% probability of the Laplace approximation to the posterior can.  Therefore
% we can calculate the approximate log marginal likelihood of the data 
% without resorting to Monte Carlo integration
%
% David Pfau, 2012

if nargin == 2 % Alternative calling convention for evaluating held-out data: data_log_lik( data, params )
    params = map;
    [map,~,prec] = newtons_method(@(x) log_lik( data, x, params ), repmat(params.x0,1,size(data,2)), 1e-8 );
end
ll = log_lik( data, map, params ) - ...
    1/2*( map(:)'*hess_mult( prec, map(:) ) - log_det_tridiag( prec ) + numel( map )*log( 2*pi ) );