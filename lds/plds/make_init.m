function [map params] = make_init( data, k )
% For now, a random initialization of a Poisson-LDS model.  Definitely
% should implement more justified initialization later

A = eye(k);

cQ = 0.01*eye(k);
map = zeros(k,size(data,2));
for i = 2:size(data,2)
    map(:,i) = cQ*randn(k,1) + map(:,i-1);
end
Q = cQ*cQ';

Cb = newtons_method( @(x) C_likelihood( data, map, x ), zeros( size(data,1), k+1 ), 1e-8 );
f = @(x,y) exp(x);

params = struct('A',A,'C',Cb(:,1:k),'Q',Q,'b',Cb(:,end),'f',f);

function [fx,grad,H] = C_likelihood( data, map, Cb )

m = size(Cb,1);
n = size(Cb,2);
map1 = [map;ones(1,size(map,2))];
Cxb = Cb*map1;
fx = sum( sum( exp( Cxb ) - data.*Cxb ) );
grad = ( exp( Cxb ) - data )*map1';
Hinfo = tprod( tprod( exp( Cxb ), [1 3], map1, [2 3] ), [1 2 -1], map1, [3 -1] );
H = sparse( (1:m*n)'*ones(1,n), repmat((1:m)',1,n^2) + kron(0:m:(n-1)*m,ones(m,n)), Hinfo(:) );

% function test_C_likelihood( data, map, Cb )
% 
% [fx,grad] = C_likelihood( data, map, Cb );
% for i = 1:numel(Cb)
%     Cb(i) = Cb(i) + 1e-8;
%     fx_ = C_likelihood( data, map, Cb );
%     fprintf('Exact gradient: %d, Approximate gradient: %d\n',grad(i),(fx_-fx)*1e8);
%     Cb(i) = Cb(i) - 1e-8;
% end