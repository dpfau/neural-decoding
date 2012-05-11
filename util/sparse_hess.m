function spH = sparse_hess( Hinfo )
% Takes the struct of information about the Hessian and turns it into a
% sparse matrix

if isfield( Hinfo, 'diag_center' )
    N = size(Hinfo.diag_center,2);
    k = size(Hinfo.diag_left,1);
    
    i_diag = repmat((1:k)',1,N*k) + kron(0:k:(N-1)*k,ones(k));
    j_diag = ones(k,1)*(1:N*k);
    diags = zeros(k,k,N);
    for i = 1:N
        diags(:,:,i) = Hinfo.diag_left*diag(Hinfo.diag_center(:,i))*Hinfo.diag_right;
    end
    s_diag = [Hinfo.diag_corner, zeros(k,(N-1)*k)] + ...
             [repmat( Hinfo.diag_upper, 1, N-1 ), zeros(k)] + ...
             [zeros(k), repmat( Hinfo.diag_lower, 1, N-1 )] + ...
             reshape( diags, k, N*k );
    
    i_off_diag = i_diag(:,1:(N-1)*k);
    j_off_diag = j_diag(:,k + (1:(N-1)*k));
    s_off_diag = repmat( Hinfo.off_diag, 1, N-1 );
    
    spH = sparse( [i_diag, i_off_diag, j_off_diag], ...
                  [j_diag, j_off_diag, i_off_diag], ...
                  [s_diag, s_off_diag, s_off_diag] );
elseif isfield( Hinfo, 'all' )
    N = size(Hinfo.all,1);
    k = size(Hinfo.all,2);
    i = repmat(reshape(1:N*k,N,k),[1,1,k]);
    j = permute(i,[1 3 2]);
    spH = sparse(i(:),j(:),Hinfo.all(:));
else
    N = length(Hinfo.diag);
    spH = sparse( [1:N,1:N-1,2:N], [1:N,2:N,1:N-1], [Hinfo.diag, Hinfo.off_diag, Hinfo.off_diag] );
end