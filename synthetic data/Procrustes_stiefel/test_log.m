%%
% Test on the Riemannian zero order methods
% Problem: Procrustes problem, min_x \|Ax-B\|_F^2, x in St(n, p)
% Manifold: Stiefel St(n, p)
% Methods: Riemannian GD; Riemannian ZGD
clc; clear; close all
%% Problem Generating
n = 15; p = 5; l = 10;
K = n*p; % number of sample
A = randn(l, n);
% Generate X in Stiefel manifold
X0 = randn(n, p);
X0 = orth(X0);
% E = randn(l, p)*10e-8;
E = zeros(l, p);
B = A*X0 + E;


%% Algorithm

epsilon = 10e-3; gamma = 0.5; delta = 0.01; alpha = 1e-2;
h = alpha; % h <= K/n;
N = 5000; 
mu = 1e-8; % mu <= epsilon^2/n^(3/2)

log_norm_x = zeros(100, 1);
log_norm_z = log_norm_x;
for k=1:100
    % initialization
    X = randn(n, p);
    P = proj(X0,X+X0,n); % Projection onto the tangent space
    X = retr(X0,P,n,p);
    Z = X;
    
    norm_x = zeros(N,1);
    norm_z = norm_x;

    norm_x(1) = norm(proj(X,X+nablaf(X,A,B),n),'fro');
    norm_z(1) = norm(proj(Z,Z+nablaf(Z,A,B),n),'fro');
    
    fprintf("No. of experiment: %d\n", k);
    i = 1;
    while norm_x(i) >= epsilon && i < N
        i = i + 1;
        
        u = randn(n, p, K);
        gx = zeros(n,p);
        for j=1:K
            temp = proj(X,X+u(:,:,j),n);
            gx = gx + (f(retr(X,mu*temp,n,p),A,B)-f(X,A,B))*temp/mu;
        end
        gx = gx / K;
    %     tau = 1;
    %     while f(retr(X,-tau*gx,n,p),A,B) > f(X,A,B) - delta*tau*(norm(gx,'fro'))^2 
    %         tau = gamma * tau;
    %     end
    %     X = retr(X,-tau*gx,n,p);

        X = retr(X,-h*gx,n,p);

    %     norm(gx-proj(Z,Z+nablaf(Z,A,B),n)/norm(proj(Z,Z+nablaf(Z,A,B),n),'fro'),'fro')
    
        % value update
        norm_x(i) = norm(proj(X,X+nablaf(X,A,B),n),'fro');
        fprintf('iter: %d, norm of grad:%f\n', i, norm_x(i));
    end
    log_norm_x(k) = i;
    
    i = 1;
    while norm_z(i) >= epsilon && i < N
        i = i + 1;

        gz = proj(Z,Z+nablaf(Z,A,B),n);
    %     alpha = 1e-2;
    %     while f(retr(Z,-alpha*gz,n,p),A,B) > f(Z,A,B) - delta*alpha*(norm(gz,'fro'))^2 
    %         alpha = gamma * alpha;
    %     end
        Z = retr(Z,-alpha*gz,n,p);
        
        % value update
        norm_z(i) = norm(proj(Z,Z+nablaf(Z,A,B),n),'fro');
        fprintf('iter: %d, norm of grad:%f\n', i, norm_z(i));
    end
    log_norm_z(k) = i;
end


%% Plots
% figure;
% semilogy(value_x,'b*'); hold on;
% %semilogy(value_y,'g--'); hold on;
% semilogy(value_z,'r-o');
% title('function value');
% 
% figure;
% semilogy(dist_x,'b*'); hold on;
% %semilogy(dist_y,'g--'); hold on;
% semilogy(dist_z,'r-o');
% title('Distance toward the true point');


figure;
semilogy(norm_x,'b*'); hold on;
%semilogy(dist_y,'g--'); hold on;
semilogy(norm_z,'r-o');
title('Norm of current gradient');