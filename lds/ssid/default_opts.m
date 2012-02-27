function opts = default_opts( opts )
% Set default values for opts struct for all versions of SSID

if ~isfield( opts, 'noise' ),       opts.noise = 'none';    end
if ~isfield( opts, 'proj' ),        opts.proj = 'orth_svd'; end
if ~isfield( opts, 'tol' ),         opts.tol = 1e-3;        end
if ~isfield( opts, 'maxOrder' ),    opts.maxOrder = 10;     end
if ~isfield( opts, 'instant' ),     opts.instant = 1;       end
if ~strcmpi( opts.noise, 'none' )
    if ~isfield( opts, 'vsig' ),    opts.vsig = 1;          end
    if strcmpi( opts.noise, 'poiss' )
        if ~isfield( opts, 'rho' ),  opts.rho = 1;           end
    end
end
if ~isfield( opts, 'tfocs_path' )
    opts.tfocs_path = '/Users/davidpfau/Documents/MATLAB/TFOCS'; 
end