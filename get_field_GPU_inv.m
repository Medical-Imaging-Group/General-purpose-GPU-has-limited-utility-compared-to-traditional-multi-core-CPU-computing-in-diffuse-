
function [t11,phi,R]=get_field_GPU_inv(Mass,qvec,R);
% Calculates the field, given the Mass matrix and RHS source
% vector.
% H Dehghani Dartmouth College 2004

[nnodes,nsource]=size(qvec);
phi=zeros(nnodes,nsource);
msg=[];
flag = 0;


if isreal(Mass)
    
    
    if (nnodes) >= 3800
        if nargin == 2
            R = sparse(diag(diag(full(Mass))));
        elseif nargin == 3
            % do nothing
        end
        t0 = clock;
        for i = 1 : nsource
            x = viennacl_bicgstab_precon(Mass,full(qvec(:,i)),R);
            msg = [msg flag];
            phi(:,i) = x;
        end
        t1 = clock;%t11 = toc;
        t11 = etime(t1,t0);
        clear x;
    else
        if nargin == 2
            R = [];
            
        elseif nargin == 3
            % do nothing
        end
        t0 = clock;
        phi = culasv(single(full(Mass)),single(full(qvec)));
        t1 = clock;%t11 = toc;
        t11 = etime(t1,t0);
        % disp('Get-Field: Back Slash');
    end
    
else
    
    if (nnodes) >= 3800
        
        t0 = clock;
        Mass_real = real(Mass);
        Mass_imag = imag(Mass);
        qvec_real = real(qvec);
        qvec_imag = imag(qvec);
        clear Mass;
        Mass = sparse( [Mass_real , -Mass_imag; Mass_imag, Mass_real ]);
        clear Mass_real  Mass_imag
        if nargin == 2
          % create the precon. precon used is diag of mass, as the matrix is complex and symetry is lost if cholesky is used
          %  disp('before precon set up');
            R = Mass;
            R(:,:) = 0;
            d = Mass(1:(2*nnodes) + 1:end);
            R(1:(2*nnodes) + 1:end) = d;
            clear d;
         %   disp('after precon set up');
            
        elseif nargin == 3
            %	%%% do nothing
        end
        
        
        qvec = [ qvec_real ; qvec_imag ];
        clear qvec_real  qvec_imag
        
        
        for i = 1 : nsource
            x = viennacl_bicgstab_precon(Mass,full(qvec(:,i)),R);%use tolerance = 22 in the mex file
            msg = [msg flag];
            p(:,i) = double(x);
        end
        t1 = clock;%t11 = toc;
        t11 = etime(t1,t0);
        clear x;
        phi = complex(p(1:end/2,:),p(end/2+1:end,:));
        clear p;
    else
        
        if nargin == 2
            R = [];
            
        elseif nargin == 3
            % do nothing
        end
        Mass=single(full(Mass));
        qvec = single(full(qvec));
        t0 = clock;
        phi = culaCsv_t(Mass,qvec);
        t1 = clock;%t11 = toc;
        t11 = etime(t1,t0);
    end
    
end

if isempty(msg)
    %   disp('Used backslash!')
elseif any(msg==1)
    disp('some solutions did not converge')
elseif any(msg==2)
    disp('some solutions are unusable')
elseif any(msg==3)
    disp('some solutions from stagnated iterations')
end

phi = double(phi);
