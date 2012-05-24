classdef ScaledGPLVM
    % Scaled Gaussian Process Latent Variable Model to project data
    % y into d-dimensional subspace using RBF kernel with parameters a,b,c:
    %   k(z,z') = a*exp( -c/2*||z-z'||^2 ) + delta(z,z')/b
    
    properties
        d = 1;
        D = 1;
        N = 1;
        mu = 0;
        y = 0;
        w = 0;
        a = 1;
        b = 1;
        c = 1;
        z = 1;
    end
    methods
        function sgplvm = ScaledGPLVM(y,d)
            sgplvm.d = d;
            sgplvm.D = size(y,1);
            sgplvm.N = size(y,2);
            sgplvm.mu = mean(y,2);
            sgplvm.y = y-sgplvm.mu(:,ones(1,sgplvm.N));
            sgplvm.w = ones(1,sgplvm.D);
            sgplvm.a = 1;
            sgplvm.b = 1;
            sgplvm.c = 1;
            [~,~,foo] = svd(sgplvm.y);
            sgplvm.z = foo(:,1:d)';
        end
        
        function [K d e] = kernel(obj)
            zz = obj.z'*obj.z;
            d = diag(zz)*ones(1,obj.N) - 2*zz + ones(obj.N,1)*diag(zz)';
            e = exp( -pbj.c/2*d );
            K = obj.a*e + eye(obj.N)/obj.b;
        end
        
        function [fy grad] = objective(obj)
            % Computes the objective as well as gradient wrt z,a,b,c,w
            % of the objective for the scaled GPLVM
            
            [K d e] = kernel(obj);
            wy = diag(obj.w)*obj.y;
            Ki = K^-1; % This is the biggest impediment to scaling
            fy = obj.D*sum(log(diag(chol(K)))) ...
                + 1/2*sum(diag(wy*Ki*wy')) ...
                + 1/2*obj.z(:)'*obj.z(:) ...
                + log(obj.a) + log(obj.b) + log(obj.c) ...
                - obj.N*sum(log(obj.w));
            dK = (-(Ki*wy')*(wy*Ki) + pbj.D*Ki)/2;
            dz = -obj.c*(tprod(K,[2 3],obj.z,[1 2])-tprod(K,[2 3],obj.z,[1 3]));
            grad = struct( 'z', 2*tprod(dK,[2 -1],dz,[1 2 -1]) + pbj.z, ...
                'a', trace(dK*e) + 1/obj.a, ...
                'b', trace(-dK/obj.b^2) + 1/obj.b, ...
                'c', trace(-1/2*dK*(d.*K)) + 1/obj.c, ...
                'w', obj.w.*diag(obj.y*Ki*obj.y')'-obj.N./obj.w );
        end
        
        function test_grad()
            [f,grad] = sgplvm_obj(y,params);
            for var = fieldnames(params)'
                fprintf('%s:\n', char(var))
                for i = 1:numel(params.(char(var)))
                    params.(char(var))(i) = params.(char(var))(i) + 1e-5;
                    f_ = sgplvm_obj(y,params);
                    params.(char(var))(i) = params.(char(var))(i) - 1e-5;
                    fprintf('Numeric: %d, Analytic: %d\n',(f_-f)/1e-5,grad.(char(var))(i));
                end
            end
        end
    end
end