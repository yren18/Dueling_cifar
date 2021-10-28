%%
% Test on the Riemannian zero order methods
% Compare with zeroth-order projected gradient
% Problem: Procrustes problem, min_x \|Ax-B\|_F^2, x in St(n, p)
% Manifold: Stiefel St(n, p)
% Methods: Riemannian GD; Riemannian ZGD
clc; clear; close all
%% Problem Generating
n = 50; p = 25; l = 30;
K = 200; % number of sample
A = randn(l, n);
% Generate X in Stiefel manifold
X0 = randn(n, p);
X0 = orth(X0);
% E = randn(l, p)*10e-5;
% B = A*X0 + E;
B = A*X0;

%% Algorithm
% initialization
X = randn(n, p);
P = proj(X0,X+X0,n); % Projection onto the tangent space
X = retr(X0,P,n,p);
Y = X; Z = X;

epsilon = 10e-2; gamma = 0.5; delta = 0.01; alpha = 1e-3;
h = alpha;% h = 1/(n*norm(A)^2);
N = 1000; mu = 1/n^2*epsilon;
value_x = zeros(N,1);
value_y = value_x; 
value_z = value_x;

dist_x = zeros(N,1);
dist_y = dist_x; 
dist_z = dist_x;

norm_x = zeros(N,1);
norm_y = norm_x; 
norm_z = norm_x;
for i=1:N
    % value update
    value_x(i) = f(X,A,B);
    value_y(i) = f(Y,A,B);
    value_z(i) = f(Z,A,B);
    
    dist_x(i) = norm(X-X0,'fro');
    dist_y(i) = norm(Y-X0,'fro');
    dist_z(i) = norm(Z-X0,'fro');
    
    norm_x(i) = norm(proj(X,X+nablaf(X,A,B),n),'fro');
    norm_y(i) = norm(proj(Y,Y+nablaf(Y,A,B),n),'fro');
    norm_z(i) = norm(proj(Z,Z+nablaf(Z,A,B),n),'fro');
    fprintf('iter: %d, function val: %f, dist: %f, norm of grad:%f\n', i, value_x(i), dist_x(i), norm_x(i));
    %% Design #1
    u = randn(n, p, K);
    gx = zeros(n,p);
    for j=1:K
        temp = proj(X,X+u(:,:,j),n);
        gx = gx + (f(retr(X,mu*temp,n,p),A,B)-f(X,A,B))*temp/mu;
    end
    gx = gx / K;
%     alpha = 1;
%     while f(retr(X,-alpha*gx,n,p),A,B) > f(X,A,B) - delta*alpha*(norm(gx,'fro'))^2 
%         alpha = gamma * alpha;
%     end
%     X = retr(X,-alpha*gx,n,p);
    
    X = retr(X,-h*gx,n,p);
     
%     norm(gx-proj(Z,Z+nablaf(Z,A,B),n)/norm(proj(Z,Z+nablaf(Z,A,B),n),'fro'),'fro')

    %% Projected GD
    gy = nablaf(Y,A,B);
    Y = retr(Y,-alpha*gy,n,p);
    
    %% Riemannian GD
    gz = proj(Z,Z+nablaf(Z,A,B),n);
%     alpha = 1e-2;
%     while f(retr(Z,-alpha*gz,n,p),A,B) > f(Z,A,B) - delta*alpha*(norm(gz,'fro'))^2 
%         alpha = gamma * alpha;
%     end
    Z = retr(Z,-alpha*gz,n,p);
end

%% Plots
figure;
semilogy(value_x,'b*'); hold on;
semilogy(value_y,'go'); hold on;
semilogy(value_z,'r-o');
title('function value');

figure;
semilogy(dist_x,'b*'); hold on;
semilogy(dist_y,'go'); hold on;
semilogy(dist_z,'r-o');
title('Distance toward the true point');


figure;
semilogy(norm_x,'b*'); hold on;
semilogy(norm_y,'go'); hold on;
semilogy(norm_z,'r-o');
title('Norm of current gradient');