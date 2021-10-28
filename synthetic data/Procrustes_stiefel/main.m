%%
% Test on the Riemannian zero order methods
% Problem: Procrustes problem, min_x \|Ax-B\|_F^2, x in St(n, p)
% Manifold: Stiefel St(n, p)
% Methods: Riemannian GD; Riemannian ZGD
clc; clear; close all
%% Problem Generating
n = 15; p = 5; l = 10;
K = n*p - p*(p+1)/2; % number of sample
A = randn(l, n);
% Generate X in Stiefel manifold
X0 = randn(n, p);
X0 = orth(X0);
% E = randn(l, p)*10e-8;
E = zeros(l, p);
B = A*X0 + E;


%% Algorithm
% initialization
X = randn(n, p);
P = proj(X0,X+X0,n); % Projection onto the tangent space
X = retr(X0,P,n,p);
Y = X; Z = X;

epsilon = 1e-2;
alpha = 1e-2; h = 1e-2; % step-size
N = 300; 
mu = 1e-8; % mu <= epsilon^2/n^(3/2)
value_x = zeros(N,1);
value_z = value_x;

dist_x = zeros(N,1);
dist_z = dist_x;

norm_x = zeros(N,1);
norm_z = norm_x;

time_x = zeros(N,1);
time_z = time_x;

start = cputime;
for i=1:N
    % value update
    value_x(i) = f(X,A,B);
    dist_x(i) = norm(X-X0,'fro');
    norm_x(i) = norm(proj(X,X+nablaf(X,A,B),n),'fro');
    % fprintf('iter: %d, function val: %f, dist: %f, norm of grad:%f\n', i, value_x(i), dist_x(i), norm_x(i));
    %% Design #1
    u = randn(n, p, K);
    gx = zeros(n,p);
    for j=1:K
        temp = proj(X,u(:,:,j),n);
        gx = gx + (f(retr(X,mu*temp,n,p),A,B)-f(X,A,B))*temp/mu;
    end
    gx = gx / K;
%     tau = 1;
%     while f(retr(X,-tau*gx,n,p),A,B) > f(X,A,B) - delta*tau*(norm(gx,'fro'))^2 
%         tau = gamma * tau;
%     end
%     X = retr(X,-tau*gx,n,p);
    
    X = retr(X,-h*gx,n,p);
    time_x(i) = cputime - start;
     
%     norm(gx-proj(Z,Z+nablaf(Z,A,B),n)/norm(proj(Z,Z+nablaf(Z,A,B),n),'fro'),'fro')
end

start = cputime;
for i=1:N
    % value update
    value_z(i) = f(Z,A,B);
    dist_z(i) = norm(Z-X0,'fro');
    norm_z(i) = norm(proj(Z,Z+nablaf(Z,A,B),n),'fro');
    % fprintf('iter: %d, function val: %f, dist: %f, norm of grad:%f\n', i, value_x(i), dist_x(i), norm_x(i));
    %% Riemannian GD
    gz = proj(Z,Z+nablaf(Z,A,B),n);
%     alpha = 1e-2;
%     while f(retr(Z,-alpha*gz,n,p),A,B) > f(Z,A,B) - delta*alpha*(norm(gz,'fro'))^2 
%         alpha = gamma * alpha;
%     end
    Z = retr(Z,-alpha*gz,n,p);
    time_z(i) = cputime - start;
end

% save the running result
filename = [num2str(n),'_',num2str(p),'_result.mat'];
save(filename,'norm_x','norm_z');

%% Plots
h1 = figure;
semilogy(value_x,'b*'); hold on;
semilogy(value_z,'r-o');
title('function value');

h2 = figure;
semilogy(dist_x,'b*'); hold on;
semilogy(dist_z,'r-o');
title('Distance toward the true point');

h3 = figure;
semilogy(norm_x,'b-*'); hold on;
semilogy(norm_z,'r-o');
title('Norm of current gradient');
legend('ZO-RGD', 'RGD')
xlabel('number of iterations')
saveas(gcf,'procruste_np_'+ string(n) + '_' + string(p) +'_m_asdim.pdf')
% saveas(gcf,'procruste_np_'+ string(n) + '_' + string(p) + '_m_'+string(K)+'.pdf')

h4 = figure;
semilogy(time_x, norm_x,'b-*'); hold on;
semilogy(time_z, norm_z,'r-o');
title('Norm of current gradient');
legend('ZO-RGD', 'RGD')
xlabel('CPU time')
saveas(gcf,'procruste_np_'+ string(n) + '_' + string(p) +'_m_asdim_time.pdf')