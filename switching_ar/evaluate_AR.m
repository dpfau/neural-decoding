function [ar_ll switching_ar_ll] = evaluate_AR( data, nTrain, nTest, nSwitch, nRestart, maxOrder, dec )

y = decimate( fill_inf( data(:,1:nTrain) ), dec );
z = decimate( fill_inf( data(:,nTrain+(1:nTest)) ), dec );
ar_ll = zeros( maxOrder, 1 );
for i = 1:maxOrder
    fprintf('Evaluating AR(%i) model...\n',i);
    [~,~,~,ar_ll(i)] = AR(y,z,i);
end

switching_ar_ll = zeros( maxOrder, nSwitch, nRestart );
for i = 1:maxOrder
    for j = 1:nSwitch
        fprintf('Evaluating switching AR(%i) model with %i states',i,j);
        for k = nRestart
            [A Q T p0 mx] = switching_AR_EM( y, i, j );
            foo = switching_AR_forward( y, A, Q, T, p0 );
            [~,c] = switching_AR_forward( z - repmat( mx, 1, size( z, 2 ) ), A, Q, T, foo(:,end) ); % Cheating slightly by initializing where training data left off
            switching_ar_ll( i, j, k ) = sum( log( c ) );
            fprintf('.');
        end
        fprintf('\n');
    end
end