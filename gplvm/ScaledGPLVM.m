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
        
        function [K d e] = kernel(obj,fa,fb,fc)
            if nargin == 1
                fa = obj.a;
                fb = obj.b;
                fc = obj.c;
            end
            zz = obj.z'*obj.z;
            d = diag(zz)*ones(1,obj.N) - 2*zz + ones(obj.N,1)*diag(zz)';
            e = exp( -fc/2*d );
            K = fa*e + eye(obj.N)/fb;
        end
        
        function [fy grad] = f(obj,params)
            % Computes the objective as well as gradient wrt z,a,b,c,w
            % of the objective for the scaled GPLVM
            
            if nargin == 2
                fa = params(1);
                fb = params(2);
                fc = params(3);
                fw = params(4:obj.D+3);
                fz = reshape(params(obj.D+4:end),obj.d,obj.N);
            else
                fa = obj.a;
                fb = obj.b;
                fc = obj.c;
                fw = obj.w;
                fz = obj.z;
            end
            [K g e] = kernel(obj,fa,fb,fc);
            wy = diag(fw)*obj.y;
            Ki = K^-1; % This is the biggest impediment to scaling
            fy = obj.D*sum(log(diag(chol(K)))) ...
                + 1/2*sum(diag(wy*Ki*wy')) ...
                + 1/2*fz(:)'*fz(:) ...
                + log(fa) + log(fb) + log(fc) ...
                - obj.N*sum(log(fw));
            dK = (-(Ki*wy')*(wy*Ki) + pbj.D*Ki)/2;
            dz1 = -fc*(tprod(K,[2 3],fz,[1 2])-tprod(K,[2 3],fz,[1 3]));
            dz = 2*tprod(dK,[2 -1],dz1,[1 2 -1]) + fz;
            dw = fw.*diag(obj.y*Ki*obj.y')'-obj.N./fw;
            grad = [ trace(dK*e) + 1/fa; ...
                     trace(-dK/fb^2) + 1/fb; ...
                     trace(-1/2*dK*(g.*K)) + 1/fc; ...
                     dw(:); ...
                     dz(:) ];
        end
        
        function obj = assign_params(obj,params)
            % Because gradients are returned in long vector form, we need
            % to be able to assign parameters from one long vector
            
            obj.a = params(1);
            obj.b = params(2);
            obj.c = params(3);
            obj.w = params(4:obj.D+3);
            obj.z = reshape(params(obj.D+4:end),obj.d,obj.N);
        end
        
        function params = get_params(obj) 
            params = [obj.a; obj.b; obj.c; obj.w(:); obj.z(:)];
        end
        
        function test_grad(obj)
            [f,grad] = obj.f();
            params = obj.get_params;
            for i = 1:length(params)
                params(i) = params(i) + 1e-5;
                f_ = obj.f(params);
                params(i) = params(i) - 1e-5;
                fprintf('Numeric: %d, Analytic: %d\n',(f_-f)/1e-5,grad(i));
            end
        end
    end
end