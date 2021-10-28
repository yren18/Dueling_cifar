%%
% Test on the Riemannian zero order methods
% Problem: sPCA, min -1/2*tr(X^THX)+\lambda*\|X\|_1=f(X)+h(X)
% Manifold: Stiefel manifold St(n, p)
% Methods: Riemannian SubGD; ManPG; Zo-ManPG
clc; clear; close all
addpath misc
addpath SSN_subproblem
%% Problem Generating
n = 100; p = 10;
K = 50; % number of real non-zero singular values
m = n*p; % number of Gaussian sample
A = randn(n, K); A = orth(A);
Lambda = diag(abs(randn(K,1)));
H = A*Lambda*A.';
lambda = 0.5; % the mu 
F = @(X) f(X,H) + lambda*h(X);

% make sure eigenvalue sorted
% [V,D] = eig(H);
% [D,I] = sort(diag(D));
% V = V(:, I);
% H = V*diag(D)*V.';

%% Algorithm
% initialization in Stiefel manifold
X = randn(n, p);
X = orth(X);
Y = X; Z = X;

% Parameters for ManPG subproblem
L = abs(eigs(full(H),1)); % Lipschitz constant
t = 1/L;
inner_iter = 100;
prox_fun = @(b,l,r) proximal_l1(b,l,r); % proximal function used in solving the subproblem
t_min = 1e-4; % minimum stepsize
num_linesearch_x = 0; num_linesearch_y = 0;
num_inexact_x = 0; num_inexact_y = 0;
inner_flag_x = 0; inner_flag_y = 0;
Dn = sparse(DuplicationM(p)); % vectorization for SSN
pDn = (Dn'*Dn)\Dn'; % for SSN
F_val(1) = F(X); Fy_val(1) = F(Y);
nu = 0.8; % penalty coefficient?

tol = 1e-8*n*p;
alpha_x = 1; % stepsize of ManPG
alpha_y = 1; % stepsize of Zo-ManPG
alpha_z = 5e-3; % stepsize of Man-SubGD
N = 100; % number of iteration
mu = 1e-8; % mu <= epsilon^2/n^(3/2)
value_z = zeros(N,1);
value_z(1) = F(Z);

norm_x = zeros(N,1);
norm_y = norm_x;
norm_z = norm_x;

step_size_x = zeros(N,1);
step_size_y = step_size_x;
step_size_z = step_size_x;

num_inner_x = zeros(N,1); num_inner_y = zeros(N,1);
opt_sub_x = zeros(N,1); opt_sub_y = zeros(N,1);

for iter=2:N+1
    %% ManPG
    neg_pgx = -nablaf(X, H);
    if alpha_x < t_min || num_inexact_x > 10
        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end
    
    % The subproblem
    if iter == 2
         [ PX,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,X,t,X + t*neg_pgx,nu*t,inner_tol,prox_fun,inner_iter,zeros(p),Dn,pDn);
        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    else
         [ PX,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,X,t,X + t*neg_pgx,nu*t,inner_tol,prox_fun,inner_iter,Lam_x,Dn,pDn);
        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    end
    
    if in_flag == 1   % subprolem not exact.
        inner_flag_x = 1 + inner_flag_x;
    end
    
    V = PX-X; % The V solved from SSN
    
    % projection onto the Stiefel manifold
    [U, SIGMA, S] = svd(PX'*PX);   SIGMA =diag(SIGMA);    X_temp = PX*(U*diag(sqrt(1./SIGMA))*S');
    
    f_trial = f(X_temp,H);
    F_trial = f_trial + lambda*h(X_temp);   normV=norm(V,'fro');

%     if  normD < tol 
%         break;
%     end
    
%     %%% linesearch
%     alpha_x = 1;
%     while F_trial >= F_val(iter-1)-0.5/t*alpha_x*normV^2
%         alpha_x = 0.5*alpha_x;
%         linesearch_flag = 1;
%         num_linesearch_x = num_linesearch_x + 1;
%         if alpha_x < t_min
%             num_inexact_x = num_inexact_x + 1;
%             break;
%         end
%         PX = X+alpha_x*V;
%         % projection onto the Stiefel manifold
%         [U, SIGMA, S] = svd(PX'*PX);   SIGMA =diag(SIGMA);   X_temp = PX*(U*diag(sqrt(1./SIGMA))*S');
%         f_trial = f(X_temp,H);
%         F_trial = f_trial + lambda*h(X_temp);
%     end
%     X = X_temp; step_size_x(iter) = alpha_x;
%     F_val(iter) = F_trial;
%     norm_x(iter) = normV;
    
    %%% Without linesearch
    PX = X+alpha_x*V;
    % projection onto the Stiefel manifold
    [U, SIGMA, S] = svd(PX'*PX);   SIGMA =diag(SIGMA);   X_temp = PX*(U*diag(sqrt(1./SIGMA))*S');
    X = X_temp; % update
    F_val(iter) = f(X_temp,H) + lambda*h(X_temp);
    norm_x(iter) = normV;
    
    %% Zo-ManPG
    neg_pgy = -zero_oracle(Y, H, mu, m);
    if alpha_y < t_min || num_inexact_y > 10
        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end
    
    % The subproblem
    if iter == 2
         [ PY,num_inner_y(iter),Lam_y, opt_sub_y(iter),in_flag] = Semi_newton_matrix(n,p,Y,t,Y + t*neg_pgy,nu*t,inner_tol,prox_fun,inner_iter,zeros(p),Dn,pDn);
        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    else
         [ PY,num_inner_y(iter),Lam_y, opt_sub_y(iter),in_flag] = Semi_newton_matrix(n,p,Y,t,Y + t*neg_pgy,nu*t,inner_tol,prox_fun,inner_iter,Lam_y,Dn,pDn);
        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    end
    
    if in_flag == 1   % subprolem not exact.
        inner_flag_y = 1 + inner_flag_y;
    end
    
    V = PY-Y; % The V solved from SSN
    
    % projection onto the Stiefel manifold
    [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);    Y_temp = PY*(U*diag(sqrt(1./SIGMA))*S');
    
    f_trial = f(Y_temp,H);
    F_trial = f_trial + lambda*h(Y_temp);   normV=norm(V,'fro');

%     if  normD < tol 
%         break;
%     end
    
%     %%% linesearch
%     alpha_y = 1;
%     while F_trial >= Fy_val(iter-1)-0.5/t*alpha_y*normV^2
%         alpha_y = 0.5*alpha_y;
%         linesearch_flag = 1;
%         num_linesearch_y = num_linesearch_y + 1;
%         if alpha_y < t_min
%             num_inexact_y = num_inexact_y + 1;
%             break;
%         end
%         PY = Y+alpha_y*V;
%         projection onto the Stiefel manifold
%         [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);   Y_temp = PY*(U*diag(sqrt(1./SIGMA))*S');
%         f_trial = f(Y_temp,H);
%         F_trial = f_trial + lambda*h(Y_temp);
%     end
%     Y = Y_temp; step_size_y(iter) = alpha_y;
%     Fy_val(iter) = F_trial;
%     norm_y(iter) = normV;
    
    %%% Without linesearch
    PY = Y+alpha_y*V;
    % projection onto the Stiefel manifold
    [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);   Y_temp = PY*(U*diag(sqrt(1./SIGMA))*S');
    Y = Y_temp;
    Fy_val(iter) = f(Y_temp,H)+lambda*h(Y_temp);
    norm_y(iter) = normV;

    %% Riemannian Subgrad method
    neg_gz = -nablaf(Z, H)-lambda*sign(Z); % negative subgradient
    neg_pgz = proj(Z,neg_gz); % projected onto the tangent space
    Z = retr(Z, alpha_z*neg_pgz);
    value_z(iter) = F(Z);
    norm_z(iter) = norm(neg_pgz,'fro');
    
    %% Value update
    fprintf('iter: %d, function val: %f, norm of grad:%f\n', iter, Fy_val(iter), norm_y(iter));
end

% save the result
filename = [num2str(n),'_',num2str(p),'_result_norm.mat'];
save(filename,'norm_x','norm_y');
filename = [num2str(n),'_',num2str(p),'_result_value.mat'];
save(filename,'F_val','Fy_val','value_z');

%% Plots
figure;
semilogy(F_val,'b*'); hold on;
semilogy(Fy_val,'g-x'); hold on;
semilogy(value_z,'r-o');
title('Function value');
legend('ManPG', 'ZO-ManPG', 'Riemannian Subgradient');

figure;
plot(norm_x(2:N+1),'b*'); hold on;
plot(norm_y(2:N+1),'g-x'); hold on;
% plot(norm_z(2:N+1),'r-o');
title('Norm of the solution of subproblem');
legend('ManPG', 'ZO-ManPG')