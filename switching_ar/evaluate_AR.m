function [ar_ll switching_ar_ll] = evaluate_AR( data, nTrain, nTest, switches, nRestart, orders, dec )

y_ = fill_inf( data(:,1:nTrain) );
z_ = fill_inf( data(:,nTrain+(1:nTest)) );
y = [];
z = [];
for i = 1:size(data,1)
    y__ = decimate( y_(i,:), dec );
    y = [y; y__];
    z__ = decimate( z_(i,:), dec );
    z = [z; z__];
end
ar_ll = zeros( numel( orders ), 1 );
for i = orders
    fprintf( 'Evaluating AR(%i) model...\n', i );
    [~,~,~,ar_ll(i)] = AR(y,z,i);
end

switching_ar_ll = zeros( numel( orders ), numel( switches ), nRestart );
% for i = 1:length(orders)
%     for j = 1:length(switches)
%         fprintf( 'Evaluating switching AR(%i) model with %i states', orders(i), switches(j) );
%         for k = 1:nRestart
%             [A Q T p0 mx] = switching_AR_EM( y, orders(i), switches(j) );
%             foo = switching_AR_forward( y, A, Q, T, p0 );
%             [~,c] = switching_AR_forward( z - repmat( mx, 1, size( z, 2 ) ), A, Q, T, foo(:,end) ); % Cheating slightly by initializing where training data left off
%             switching_ar_ll( i, j, k ) = mean( log( c ) );
%             fprintf('.');
%         end
%         fprintf('\n');
%     end
% end