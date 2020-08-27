
function [t1,fwd_mesh,pj_error] = reconstruct_stnd_GPU_inv(fwd_fn,frequency,data_fn,iteration,lambda,output_fn,filter_n)


% load fine mesh for fwd solve
fwd_mesh = load_mesh(fwd_fn);
% 
% if ischar(recon_basis)
%   recon_mesh = load_mesh(recon_basis);
%   [fwd_mesh.fine2coarse,...
%    recon_mesh.coarse2fine] = second_mesh_basis(fwd_mesh,recon_mesh);
% else
%   [fwd_mesh.fine2coarse,recon_mesh] = pixel_basis(recon_basis,fwd_mesh);
% end

% read data
anom = load_data(data_fn);
anom = anom.paa;
anom(:,1) = log(anom(:,1));
anom(:,2) = anom(:,2)/180.0*pi;
anom(find(anom(:,2)<0),2) = anom(find(anom(:,2)<0),2) + (2*pi);
anom(find(anom(:,2)>(2*pi)),2) = anom(find(anom(:,2)>(2*pi)),2) - (2*pi);
anom = reshape(anom',length(anom)*2,1);

% Initiate projection error
pj_error = [];

% Initiate log file
fid_log = fopen([output_fn '.log'],'w');
fprintf(fid_log,'Forward Mesh   = %s\n',fwd_fn);
% if ischar(recon_basis)
%   fprintf(fid_log,'Basis          = %s\n',recon_basis);
% else
%   fprintf(fid_log,'Basis          = %s\n',num2str(recon_basis));
% end
fprintf(fid_log,'Frequency      = %f MHz\n',frequency);
fprintf(fid_log,'Data File      = %s\n',data_fn);
fprintf(fid_log,'Initial Reg    = %d\n',lambda);
fprintf(fid_log,'Filter         = %d\n',filter_n);
fprintf(fid_log,'Output Files   = %s_mua.sol\n',output_fn);
fprintf(fid_log,'               = %s_mus.sol\n',output_fn);
t1 = [];

for it = 1 : iteration
  t1 = [t1 it];
    %tic;
  T0 = clock;
    
  % Calculate jacobian
  %tic;
  [t12,J,data]=jacobian_GPU_inv(fwd_mesh,frequency);
  T1 = clock;%t11 = toc
  
  t1 = [t1 t12 etime(T1,T0)]; 
  
  % Read reference data
  clear ref;
  ref(:,1) = log(data.amplitude);
  ref(:,2) = data.phase;
  ref(:,2) = ref(:,2)/180.0*pi;
  ref(find(ref(:,2)<0),2) = ref(find(ref(:,2)<0),2) + (2*pi);
  ref(find(ref(:,2)>(2*pi)),2) = ref(find(ref(:,2)>(2*pi)),2) - (2*pi);
  ref = reshape(ref',length(ref)*2,1);

  data_diff = (anom-ref);

  pj_error = [pj_error sum((anom-ref).^2)];  
  
  disp('---------------------------------');
  disp(['Iteration Number          = ' num2str(it)]);
  disp(['Projection error          = ' num2str(pj_error(end))]);

  fprintf(fid_log,'---------------------------------\n');
  fprintf(fid_log,'Iteration Number          = %d\n',it);
  fprintf(fid_log,'Projection error          = %f\n',pj_error(end));

  if it ~= 1
    p = (pj_error(end-1)-pj_error(end))*100/pj_error(end-1);
    disp(['Projection error change   = ' num2str(p) '%']);
    fprintf(fid_log,'Projection error change   = %f %%\n',p);
    if (p) <= 2
      disp('---------------------------------');
      disp('STOPPING CRITERIA REACHED');
      fprintf(fid_log,'---------------------------------\n');
      fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
     break
    end
  end

  % Interpolate onto recon mesh
  %[J,recon_mesh] = interpolatef2r(fwd_mesh,recon_mesh,J.complete);
   J = J.complete;
    
  % Normalize Jacobian wrt optical values
 N = [fwd_mesh.kappa fwd_mesh.mua];
  nn = length(fwd_mesh.nodes);
  % Normalise by looping through each node, rather than creating a
  % diagonal matrix and then multiplying - more efficient for large meshes
  for i = 1 : nn
      J(:,i) = J(:,i).*N(i,1);
      J(:,i+nn) = J(:,i+nn).*N(i,2);
  end
    %J = J*diag([fwd_mesh.kappa;fwd_mesh.mua]);
clear N;
  
   [nrow,ncol]=size(J);
   
   % Add regularization
  if it ~= 1
    lambda = lambda./10^0.25;
  end
  
  
  
  
  
  
  reg = ones(nrow,1);
  
   reg_amp = lambda*(max(max(abs(J(1:2:end,:))))^2);
  reg_phs = lambda*(max(max(abs(J(2:2:end,:))))^2);
  reg(1:2:end) = reg(1:2:end).*reg_amp;
  reg(2:2:end) = reg(2:2:end).*reg_phs;
  reg_matrix = diag(reg);
  
  clear reg;
   disp(['Amp Regularization        = ' num2str(reg_matrix(1,1))]);
  disp(['Phs Regularization        = ' num2str(reg_matrix(2,2))]);
  fprintf(fid_log,'Amp Regularization        = %f\n',reg_matrix(1,1));
  fprintf(fid_log,'Phs Regularization        = %f\n',reg_matrix(2,2));
  
  disp('Calculating Update GPU')
 
  % calculate the update
  T2 = clock;
  foo = calc_update_gpu_JTJ(single(J), single(reg_matrix), single(data_diff));
  %foo = calc_update_gpu_JJT(single(J), single(reg_matrix), single(data_diff));
  T1 = clock;%t11 = toc
  
  t1 = [t1 etime(T1,T2)]; 
  
  
  
  %t11 = toc;
  %t1 = [t1 t11];
  clear J data_diff reg_matrix;  
  foo = double(foo);
  
%   % build hessian
%   [nrow,ncol]=size(J);
%   Hess = zeros(nrow);
%   Hess = (J*J');
%   
%   % initailize temp Hess, data and mesh, incase PJ increases.
%   Hess_tmp = Hess;
%   mesh_tmp = recon_mesh;
%   data_tmp = data_diff;
  


  % Seems that scatter part is always more noisey. So we will make
  % sure that the regularization for phase is some factor (ratio
  % of amplitude vs phase diagonals of the Hessian) higher.
  % This 1st method not as objective as the second and implemented
  % method.
  %
  % reg = lambda*max(diag(Hess));
  % ph_factor = median(diag(Hess(1:2:end,1:2:end)) ./ ...
  %		       diag(Hess(2:2:end,2:2:end)));
  % reg = ones(nrow,1).*reg;
  % reg(1:2:end) =  reg(1:2:end).*ph_factor;
  % reg = diag(reg);
  
%   reg_amp = lambda*max(diag(Hess(1:2:end,1:2:end)));
%   reg_phs = lambda*max(diag(Hess(2:2:end,2:2:end)));
%   reg = ones(nrow,1);
%   reg(1:2:end) = reg(1:2:end).*reg_amp;
%   reg(2:2:end) = reg(2:2:end).*reg_phs;
%   reg = diag(reg);
%   
%  
%   Hess = Hess+reg;

  % Calculate update
  %foo = J'*(Hess\data_diff);
  foo = foo.*[fwd_mesh.kappa;fwd_mesh.mua];
 
  % Update values

  fwd_mesh.kappa = fwd_mesh.kappa + foo(1:end/2);
  fwd_mesh.mua = fwd_mesh.mua + (foo(end/2+1:end));
  fwd_mesh.mus = (1./(3.*fwd_mesh.kappa))-fwd_mesh.mua;
  
   T1 = clock;%t11 = toc
  
  t1 = [t1 etime(T1,T0)]; 
  
  
 clear foo

  % Interpolate optical properties to fine mesh
  %[fwd_mesh,recon_mesh] = interpolatep2f(fwd_mesh,recon_mesh);

  % We dont like -ve mua or mus! so if this happens, terminate
  if (any(fwd_mesh.mua<0) | any(fwd_mesh.mus<0))
    disp('---------------------------------');
    disp('-ve mua or mus calculated...not saving solution');
    fprintf(fid_log,'---------------------------------\n');
    fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
    break
  end
  
  % Filtering if needed!
  if filter_n > 1
    fwd_mesh = mean_filter(fwd_mesh,abs(filter_n));
  elseif filter_n < 1
    fwd_mesh = median_filter(fwd_mesh,abs(filter_n));
  end

  if it == 1
    fid = fopen([output_fn '_mua.sol'],'w');
  else
    fid = fopen([output_fn '_mua.sol'],'a');
  end
  fprintf(fid,'solution %g ',it);
  fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
  fprintf(fid,'-components=1 ');
  fprintf(fid,'-type=nodal\n');
  fprintf(fid,'%f ',fwd_mesh.mua);
  fprintf(fid,'\n');
  fclose(fid);
  
  if it == 1
    fid = fopen([output_fn '_mus.sol'],'w');
  else
    fid = fopen([output_fn '_mus.sol'],'a');
  end
  fprintf(fid,'solution %g ',it);
  fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
  fprintf(fid,'-components=1 ');
  fprintf(fid,'-type=nodal\n');
  fprintf(fid,'%f ',fwd_mesh.mus);
  fprintf(fid,'\n');
  fclose(fid);
end
%plotimage(fwd_mesh,fwd_mesh.mus);
% close log file!
%fprintf(fid_log,'Computation Time = %f\n',time);
fclose(fid_log);





function [val_int,recon_mesh] = interpolatef2r(fwd_mesh,recon_mesh,val)

% This function interpolates fwd_mesh into recon_mesh
% For the Jacobian it is an integration!
NNC = size(recon_mesh.nodes,1);
NNF = size(fwd_mesh.nodes,1);
NROW = size(val,1);
val_int = zeros(NROW,NNC*2);

for i = 1 : NNF
  if recon_mesh.coarse2fine(i,1) ~= 0
    val_int(:,recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)) = ...
    val_int(:,recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)) + ...
	val(:,i)*recon_mesh.coarse2fine(i,2:end);
    val_int(:,recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)+NNC) = ...
    val_int(:,recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)+NNC) + ...
	val(:,i+NNF)*recon_mesh.coarse2fine(i,2:end);
  elseif recon_mesh.coarse2fine(i,1) == 0
    dist = distance(fwd_mesh.nodes,fwd_mesh.bndvtx,recon_mesh.nodes(i,:));
    mindist = find(dist==min(dist));
    mindist = mindist(1);
    val_int(:,i) = val(:,mindist);
    val_int(:,i+NNC) = val(:,mindist+NNF);
  end
end

for i = 1 : NNC
  if fwd_mesh.fine2coarse(i,1) ~= 0
    recon_mesh.mua(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
    fwd_mesh.mua(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    recon_mesh.mus(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
    fwd_mesh.mus(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    recon_mesh.kappa(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
    fwd_mesh.kappa(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    recon_mesh.region(i,1) = ...
    median(fwd_mesh.region(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
  elseif fwd_mesh.fine2coarse(i,1) == 0
    dist = distance(fwd_mesh.nodes,...
		    fwd_mesh.bndvtx,...
		    [recon_mesh.nodes(i,1:2) 0]);
    mindist = find(dist==min(dist));
    mindist = mindist(1);
    recon_mesh.mua(i,1) = fwd_mesh.mua(mindist);
    recon_mesh.mus(i,1) = fwd_mesh.mus(mindist);
    recon_mesh.kappa(i,1) = fwd_mesh.kappa(mindist);
    recon_mesh.region(i,1) = fwd_mesh.region(mindist);
  end
end

function [fwd_mesh,recon_mesh] = interpolatep2f(fwd_mesh,recon_mesh)


for i = 1 : length(fwd_mesh.nodes)
  fwd_mesh.mua(i,1) = ...
      (recon_mesh.coarse2fine(i,2:end) * ...
       recon_mesh.mua(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
  fwd_mesh.kappa(i,1) = ...
      (recon_mesh.coarse2fine(i,2:end) * ...
       recon_mesh.kappa(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
  fwd_mesh.mus(i,1) = ...
      (recon_mesh.coarse2fine(i,2:end) * ...
       recon_mesh.mus(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
end


