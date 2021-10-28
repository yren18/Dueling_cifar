%% Test on the Riemannian GD
% Problem: Procrustes problem, min_x \|Ax-B\|_F^2, x in St(n, p)
clc; clear; close all
%% Problem Generating
n = 500; p = 300; l = 200;
A = randn(l, n);
% Generate X in Stiefel manifold
[X0,~]=svd(randn(n,p),0);
E = randn(l, p)*10e-5;
B = A*X0 + E;

%% Algorithm
% initialization
X = randn(n, p);
P = proj(X0,X+X0,n); % Projection onto the tangent space
X = retr(X0,P,n,p);

epsilon = 10e-2;
gamma = 0.5; delta = 0.1;
N = 100; h = 1/(n*norm(A)^2); mu = 1/n^2;
value = zeros(N,1);
dist = zeros(N,1);
for i=1:N
    value(i) = f(X,A,B);
    dist(i) = norm(X-X0,'fro');
    
    % Calculate the Riemannian gradient
    g = proj(X,X+nablaf(X,A,B),n);
    % g = g / norm(g,'fro');
    
    % The line search process
    alpha = 1;
    while f(retr(X,-alpha*g,n,p),A,B) > f(X,A,B) - delta*alpha*(norm(g,'fro'))^2 
        alpha = gamma * alpha;
    end
    X = retr(X,-alpha*g,n,p);
end
figure;
plot(value,'b*');
hold on;
title('function value');

figure;
plot(dist,'b*');
hold on;
title('Distance toward the true point');