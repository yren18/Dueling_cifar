n = 50; p = 1; l = 30;
K = 1000; % number of sample
epsilon = 0.01;
mu = 1/n^2*epsilon;
A = randn(l, n);
% Generate X in Stiefel manifold
X0 = randn(n, p);
E = randn(l, p)*10e-5;
B = A*X0 + E;

X = randn(n, p);
u = randn(n, p, K);
g = zeros(n, p);
for j=1:K
    g = g + (f(X+mu*u(:,:,j),A,B)-f(X,A,B))*u(:,:,j)/mu;
end
g = g / K; 

nabla = nablaf(X,A,B);
norm(g)
norm(nabla)
norm(g-nabla)
norm(g/norm(g)-nabla/norm(nabla))