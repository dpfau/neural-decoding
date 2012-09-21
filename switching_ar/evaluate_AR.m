function [train_ll test_ll] = evaluate_AR( data, nTrain, nTest, switches, nRestart, orders, dec )

y_ = data(:,1:nTrain);
z_ = data(:,nTrain+(1:nTest));
y = [];
z = [];
for i = 1:size(data,1)
    y__ = decimate( y_(i,:), dec );
    y = [y; y__];
    z__ = decimate( z_(i,:), dec );
    z = [z; z__];
end
% ar_ll = zeros( numel( orders ), 1 );
% for i = orders
%     fprintf( 'Evaluating AR(%i) model...\n', i );
%     [~,~,~,ar_ll(i)] = AR(y,z,i);
% end

train_ll = zeros( numel( orders ), numel( switches ), nRestart );
test_ll  = zeros( numel( orders ), numel( switches ), nRestart );
for i = 1:length(orders)
    for j = 1:length(switches)
        fprintf( 'Evaluating switching AR(%i) model with %i states', orders(i), switches(j) );
        for k = 1:nRestart
            [A Q T p0 mx] = switching_AR_EM( y, orders(i), switches(j), 1 );
            foo = switching_AR_forward( y, A, Q, T, p0, mx );
            [~,c1] = switching_AR_forward( y, A, Q, T, p0, mx );
            [~,c2] = switching_AR_forward( z, A, Q, T, foo(:,end), mx ); % Cheating slightly by initializing where training data left off
            train_ll( i, j, k ) = mean( log( c1 ) );
            test_ll( i, j, k )  = mean( log( c2 ) );
            fprintf('.');
        end
        fprintf('\n');
    end
end